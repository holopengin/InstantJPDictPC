use crate::models::BoundingBox;

/// For each node (global char index): [north, south, east, west] target indices.
#[derive(Clone, Debug)]
pub struct NavGraph {
    pub edges: Vec<[usize; 4]>,
    pub initial_edges: Vec<[usize; 4]>,
    pub positions: Vec<(f32, f32)>,
    pub n: usize,
}

impl NavGraph {
    pub fn build(boxes: &[BoundingBox]) -> Self {
        let n = boxes.len();
        if n < 5 { return Self::fallback(n); }

        let mut positions = Vec::with_capacity(n);
        let (max_x, max_y) = if n > 0 {
            let mx = boxes.iter().map(|b| b.right() as f32).fold(0f32, f32::max);
            let my = boxes.iter().map(|b| b.bottom() as f32).fold(0f32, f32::max);
            (mx.max(1.0), my.max(1.0))
        } else {
            (1.0, 1.0)
        };
        for b in boxes {
            let cx = (b.left() as f32 + (b.w as f32) / 2.0) / max_x;
            let cy = (b.top() as f32 + (b.h as f32) / 2.0) / max_y;
            positions.push((cx, cy));
        }

        let mut north_lists: Vec<Vec<(usize, f32)>> = Vec::with_capacity(n);
        let mut south_lists: Vec<Vec<(usize, f32)>> = Vec::with_capacity(n);
        let mut east_lists: Vec<Vec<(usize, f32)>> = Vec::with_capacity(n);
        let mut west_lists: Vec<Vec<(usize, f32)>> = Vec::with_capacity(n);
        const W: f32 = 10.0; // off‑axis penalty weight (Phase 1 — strict local)
        const W2: f32 = 3.0; // off‑axis penalty weight (Phase 2 — island connecting)
        const LOCAL_DIST: f32 = 0.05; // max Euclidean distance for Phase 1
        const CONE45: f32 = 1.0; // allow within 45° of cardinal (off_axis ≤ primary)

        for i in 0..n {
            let (xi, yi) = positions[i];
            let mut north: Vec<(usize, f32)> = Vec::new();
            let mut south: Vec<(usize, f32)> = Vec::new();
            let mut east: Vec<(usize, f32)> = Vec::new();
            let mut west: Vec<(usize, f32)> = Vec::new();

            for j in 0..n {
                if i == j { continue; }
                let (xj, yj) = positions[j];
                let xd = (xi - xj).abs();
                let yd = (yi - yj).abs();
                let euc = (xd * xd + yd * yd).sqrt();
                if euc > LOCAL_DIST { continue; }
                // Phase 1 — strict local: cost = primary + w * off_axis, 45° cone
                let dy_n = yi - yj;
                if yj < yi && xd <= CONE45 * dy_n { north.push((j, dy_n + W * xd)); }
                let dy_s = yj - yi;
                if yj > yi && xd <= CONE45 * dy_s { south.push((j, dy_s + W * xd)); }
                let dx_e = xj - xi;
                if xj > xi && yd <= CONE45 * dx_e { east.push((j, dx_e + W * yd)); }
                let dx_w = xi - xj;
                if xj < xi && yd <= CONE45 * dx_w { west.push((j, dx_w + W * yd)); }
            }

            let sort_fn = |a: &(usize, f32), b: &(usize, f32)| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| {
                        let (xa, ya) = positions[a.0];
                        let (xb, yb) = positions[b.0];
                        let da = ((xi - xa).powi(2) + (yi - ya).powi(2)).sqrt();
                        let db = ((xi - xb).powi(2) + (yi - yb).powi(2)).sqrt();
                        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .then_with(|| a.0.cmp(&b.0))
            };
            for list in [&mut north, &mut south, &mut east, &mut west] {
                list.sort_by(sort_fn);
            }

            north_lists.push(north);
            south_lists.push(south);
            east_lists.push(east);
            west_lists.push(west);
        }

        // Phase 1: greedy local assignment
        let mut edges = greedy_assignment_all(n, &north_lists, &south_lists, &east_lists, &west_lists);
        let initial_edges = edges.clone();

        // Phase 2: global candidates (no torus, unlimited distance, 45° cone)
        let mut t_north: Vec<Vec<(usize, f32)>> = Vec::with_capacity(n);
        let mut t_south: Vec<Vec<(usize, f32)>> = Vec::with_capacity(n);
        let mut t_east: Vec<Vec<(usize, f32)>> = Vec::with_capacity(n);
        let mut t_west: Vec<Vec<(usize, f32)>> = Vec::with_capacity(n);

        for i in 0..n {
            let (xi, yi) = positions[i];
            let mut north: Vec<(usize, f32)> = Vec::new();
            let mut south: Vec<(usize, f32)> = Vec::new();
            let mut east: Vec<(usize, f32)> = Vec::new();
            let mut west: Vec<(usize, f32)> = Vec::new();

            for j in 0..n {
                if i == j { continue; }
                let (xj, yj) = positions[j];
                let xd = (xi - xj).abs();
                let yd = (yi - yj).abs();
                // Phase 2: raw screen distance, 45° cone, off‑axis penalty
                let dy_n = yi - yj;
                if yj < yi && xd <= CONE45 * dy_n { north.push((j, dy_n + W2 * xd)); }
                let dy_s = yj - yi;
                if yj > yi && xd <= CONE45 * dy_s { south.push((j, dy_s + W2 * xd)); }
                let dx_e = xj - xi;
                if xj > xi && yd <= CONE45 * dx_e { east.push((j, dx_e + W2 * yd)); }
                let dx_w = xi - xj;
                if xj < xi && yd <= CONE45 * dx_w { west.push((j, dx_w + W2 * yd)); }
            }

            let sort_fn = |a: &(usize, f32), b: &(usize, f32)| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.0.cmp(&b.0))
            };
            for list in [&mut north, &mut south, &mut east, &mut west] {
                list.sort_by(sort_fn);
            }

            t_north.push(north);
            t_south.push(south);
            t_east.push(east);
            t_west.push(west);
        }

        // Fill any Phase‑1 empty slots with best Phase‑2 candidate (no connectivity enforcement)
        for i in 0..n {
            for d in 0..4 {
                if edges[i][d] >= n {
                    let list = match d {
                        0 => &t_north[i], 1 => &t_south[i],
                        2 => &t_east[i],  _ => &t_west[i],
                    };
                    if let Some(&(v, _)) = list.first() {
                        edges[i][d] = v;
                    }
                }
            }
        }

        Self { edges, initial_edges, positions, n }
    }

    pub fn navigate(&self, idx: usize, dir: usize) -> Option<usize> {
        if idx >= self.n || dir >= 4 { return None; }
        let target = self.edges[idx][dir];
        if target >= self.n { return None; }
        Some(target)
    }

    fn fallback(n: usize) -> Self {
        let positions = vec![(0.0, 0.0); n];
        let mut edges = vec![[0usize; 4]; n];
        for i in 0..n {
            edges[i] = [(i + 1) % n, (i + 2) % n, (i + 3) % n, (i + 4) % n];
        }
        let initial_edges = edges.clone();
        Self { edges, initial_edges, positions, n }
    }
}

/// Pick best candidate per direction greedily, resolving conflicts
/// by keeping the direction whose cost increase is smallest.
fn greedy_assignment_all(
    n: usize,
    north_lists: &[Vec<(usize, f32)>],
    south_lists: &[Vec<(usize, f32)>],
    east_lists: &[Vec<(usize, f32)>],
    west_lists: &[Vec<(usize, f32)>],
) -> Vec<[usize; 4]> {
    let mut edges = vec![[0usize; 4]; n];
    for i in 0..n {
        edges[i] = greedy_assignment(i, n, &north_lists[i], &south_lists[i], &east_lists[i], &west_lists[i]);
    }
    edges
}

fn greedy_assignment(
    _i: usize, n: usize,
    north: &[(usize, f32)],
    south: &[(usize, f32)],
    east: &[(usize, f32)],
    west: &[(usize, f32)],
) -> [usize; 4] {
    // Pick top choice for each direction. Empty = n (sentinel).
    let mut result = [n; 4];
    let mut dir_candidates = [
        north.first().copied(),
        south.first().copied(),
        east.first().copied(),
        west.first().copied(),
    ];

    for d in 0..4 {
        result[d] = dir_candidates[d].map(|(idx, _)| idx).unwrap_or(n);
    }

    // Resolve conflicts: pick next best for the cheaper-to-change direction
    let mut has_conflict = true;
    while has_conflict {
        has_conflict = false;
        let mut used = std::collections::HashSet::new();
        let mut clean = true;
        for d in 0..4 {
            if result[d] == n || !used.insert(result[d]) {
                clean = false;
            }
        }
        if clean { break; }

        // Find the first conflict and resolve it
        used.clear();
        for d in 0..4 {
            if result[d] == n || !used.insert(result[d]) {
                // Conflict or empty — find best alternative
                let list = match d { 0 => north, 1 => south, 2 => east, _ => west };
                let original = result[d];
                for &(alt, _cost) in list {
                    if !used.contains(&alt) && alt != n {
                        result[d] = alt;
                        used.insert(alt);
                        has_conflict = true;
                        break;
                    }
                }
                if result[d] == original {
                    if original != n {
                        // No alternative found — pick any unused node
                        for j in 0..n {
                            if !used.contains(&j) {
                                result[d] = j;
                                used.insert(j);
                                has_conflict = true;
                                break;
                            }
                        }
                    } // else: genuinely empty direction, leave as n
                }
                break; // fix one conflict per iteration
            }
        }
    }

    result
}

fn is_strongly_connected(edges: &[[usize; 4]], n: usize) -> bool {
    if n == 0 { return true; }
    let mut visited = vec![false; n];
    let mut stack = vec![0usize];
    visited[0] = true;
    while let Some(u) = stack.pop() {
        for d in 0..4 {
            let v = edges[u][d];
            if v < n && !visited[v] { visited[v] = true; stack.push(v); }
        }
    }
    if visited.iter().any(|&v| !v) { return false; }

    let mut rev_visited = vec![false; n];
    let mut rev_stack = vec![0usize];
    rev_visited[0] = true;
    while let Some(v) = rev_stack.pop() {
        for u in 0..n {
            if !rev_visited[u] {
                for d in 0..4 {
                    if edges[u][d] == v { rev_visited[u] = true; rev_stack.push(u); break; }
                }
            }
        }
    }
    rev_visited.iter().all(|&v| v)
}

fn compute_reachable(edges: &[[usize; 4]], n: usize, start: usize) -> Vec<bool> {
    let mut visited = vec![false; n];
    let mut stack = vec![start];
    visited[start] = true;
    while let Some(u) = stack.pop() {
        for d in 0..4 {
            let v = edges[u][d];
            if v < n && !visited[v] { visited[v] = true; stack.push(v); }
        }
    }
    visited
}

fn compute_reverse_reachable(edges: &[[usize; 4]], n: usize, target: usize) -> Vec<bool> {
    let mut can_reach = vec![false; n];
    can_reach[target] = true;
    let mut changed = true;
    while changed {
        changed = false;
        for u in 0..n {
            if can_reach[u] { continue; }
            for d in 0..4 {
                if edges[u][d] < n && can_reach[edges[u][d]] {
                    can_reach[u] = true; changed = true; break;
                }
            }
        }
    }
    can_reach
}

fn candidate_cost(
    u: usize, v: usize,
    positions: &[(f32, f32)],
    candidates: &[Vec<(usize, f32)>],
) -> f32 {
    for &(c, cost) in &candidates[u] {
        if c == v { return cost; }
    }
    let (xu, yu) = positions[u];
    let (xv, yv) = positions[v];
    (xu - xv).abs() + (yu - yv).abs()
}

fn enforce_connectivity(
    edges: &mut Vec<[usize; 4]>,
    n: usize,
    positions: &[(f32, f32)],
    north_lists: &[Vec<(usize, f32)>],
    south_lists: &[Vec<(usize, f32)>],
    east_lists: &[Vec<(usize, f32)>],
    west_lists: &[Vec<(usize, f32)>],
) {
    for _iteration in 0..n {
        if is_strongly_connected(edges, n) { return; }

        let reachable = compute_reachable(edges, n, 0);
        let unreachable: Vec<usize> = (0..n).filter(|&i| !reachable[i]).collect();
        if !unreachable.is_empty() {
            let mut improved = false;
            for u in 0..n {
                if !reachable[u] { continue; }
                for d in 0..4 {
                    let old_v = edges[u][d];
                    if old_v >= n || !reachable[old_v] { continue; }
                    for &v in &unreachable {
                        if v == u || edges[u].contains(&v) { continue; }
                        let cost = match d {
                            0 => candidate_cost(u, v, positions, north_lists),
                            1 => candidate_cost(u, v, positions, south_lists),
                            2 => candidate_cost(u, v, positions, east_lists),
                            _ => candidate_cost(u, v, positions, west_lists),
                        };
                        if cost < f32::MAX {
                            edges[u][d] = v; improved = true; break;
                        }
                    }
                    if improved { break; }
                }
                if improved { break; }
            }
            if !improved { break; }
            continue;
        }

        let can_reach_root = compute_reverse_reachable(edges, n, 0);
        let cannot_reach: Vec<usize> = (0..n).filter(|&i| !can_reach_root[i]).collect();
        if !cannot_reach.is_empty() {
            let mut improved = false;
            for &u in &cannot_reach {
                let all_cant = (0..4).all(|d| !can_reach_root[edges[u][d]]);
                if all_cant {
                    for d in 0..4 {
                        for v in 0..n {
                            if v == u || edges[u].contains(&v) { continue; }
                            if can_reach_root[v] {
                                let cost = match d {
                                    0 => candidate_cost(u, v, positions, north_lists),
                                    1 => candidate_cost(u, v, positions, south_lists),
                                    2 => candidate_cost(u, v, positions, east_lists),
                                    _ => candidate_cost(u, v, positions, west_lists),
                                };
                                if cost < f32::MAX {
                                    edges[u][d] = v; improved = true; break;
                                }
                            }
                        }
                        if improved { break; }
                    }
                }
                if improved { break; }
            }
            if !improved { break; }
            continue;
        }
        break;
    }
}
