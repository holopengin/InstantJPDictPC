use crate::models::BoundingBox;

/// Torus-wrapped horizontal distance.
pub fn torus_dx(x1: f32, x2: f32) -> f32 {
    let raw = (x1 - x2).abs();
    raw.min(1.0 - raw)
}
/// Torus-wrapped vertical distance.
pub fn torus_dy(y1: f32, y2: f32) -> f32 {
    let raw = (y1 - y2).abs();
    raw.min(1.0 - raw)
}

/// For each node (global char index): [north, south, east, west] target indices.
#[derive(Clone, Debug)]
pub struct NavGraph {
    pub edges: Vec<[usize; 4]>,
    pub positions: Vec<(f32, f32)>,
    pub n: usize,
}

impl NavGraph {
    /// Build the navigation graph from character bounding boxes.
    pub fn build(boxes: &[BoundingBox]) -> Self {
        let n = boxes.len();
        if n < 5 {
            return Self::fallback(n);
        }

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

        for i in 0..n {
            let (xi, yi) = positions[i];
            let mut north: Vec<(usize, f32)> = Vec::new();
            let mut south: Vec<(usize, f32)> = Vec::new();
            let mut east: Vec<(usize, f32)> = Vec::new();
            let mut west: Vec<(usize, f32)> = Vec::new();

            for j in 0..n {
                if i == j { continue; }
                let (xj, yj) = positions[j];
                let dx = torus_dx(xi, xj);
                let dy = torus_dy(yi, yj);

                let dy_n = (yi - yj + 1.0) % 1.0;
                if dy_n > 0.0 { north.push((j, dx * 8.0 + dy_n)); }

                let dy_s = (yj - yi + 1.0) % 1.0;
                if dy_s > 0.0 { south.push((j, dx * 8.0 + dy_s)); }

                let dx_e = (xj - xi + 1.0) % 1.0;
                if dx_e > 0.0 { east.push((j, dx_e + dy * 8.0)); }

                let dx_w = (xi - xj + 1.0) % 1.0;
                if dx_w > 0.0 { west.push((j, dx_w + dy * 8.0)); }
            }

            let sort_fn = |a: &(usize, f32), b: &(usize, f32)| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            };
            north.sort_by(sort_fn);
            south.sort_by(sort_fn);
            east.sort_by(sort_fn);
            west.sort_by(sort_fn);

            north_lists.push(north);
            south_lists.push(south);
            east_lists.push(east);
            west_lists.push(west);
        }

        let mut edges = vec![[0usize; 4]; n];
        for i in 0..n {
            edges[i] = local_assignment(
                i, n,
                &north_lists[i], &south_lists[i],
                &east_lists[i], &west_lists[i],
            );
        }

        enforce_connectivity(&mut edges, n, &positions, &north_lists, &south_lists, &east_lists, &west_lists);

        Self { edges, positions, n }
    }

    /// Navigate from current global index to target in given direction (0=N,1=S,2=E,3=W).
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
            edges[i] = [
                (i + 1) % n,
                (i + 2) % n,
                (i + 3) % n,
                (i + 4) % n,
            ];
        }
        Self { edges, positions, n }
    }
}

fn local_assignment(
    i: usize, n: usize,
    north: &[(usize, f32)],
    south: &[(usize, f32)],
    east: &[(usize, f32)],
    west: &[(usize, f32)],
) -> [usize; 4] {
    let mut best = [0usize; 4];
    let mut best_cost = f32::MAX;

    for k in (4..=n.saturating_sub(1)).step_by(2).chain(std::iter::once(n.saturating_sub(1))) {
        let nk = north.len().min(k);
        let sk = south.len().min(k);
        let ek = east.len().min(k);
        let wk = west.len().min(k);

        for ni in 0..nk {
            let (nn, nc) = north[ni];
            for si in 0..sk {
                let (sn, sc) = south[si];
                if sn == nn { continue; }
                for ei in 0..ek {
                    let (en, ec) = east[ei];
                    if en == nn || en == sn { continue; }
                    for wi in 0..wk {
                        let (wn, wc) = west[wi];
                        if wn == nn || wn == sn || wn == en { continue; }
                        let cost = nc + sc + ec + wc;
                        if cost < best_cost {
                            best_cost = cost;
                            best = [nn, sn, en, wn];
                        }
                    }
                }
            }
        }
        if best_cost < f32::MAX { break; }
    }
    let mut used = std::collections::HashSet::new();
    for t in &best { if *t != 0 { used.insert(*t); } }
    for idx in 0..4 {
        if best[idx] == 0 {
            for j in 1..n {
                if !used.contains(&j) && j != i {
                    best[idx] = j;
                    used.insert(j);
                    break;
                }
            }
        }
    }
    best
}

fn is_strongly_connected(edges: &[[usize; 4]], n: usize) -> bool {
    if n == 0 { return true; }
    let mut visited = vec![false; n];
    let mut stack = vec![0usize];
    visited[0] = true;
    while let Some(u) = stack.pop() {
        for d in 0..4 {
            let v = edges[u][d];
            if v < n && !visited[v] {
                visited[v] = true;
                stack.push(v);
            }
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
                    if edges[u][d] == v {
                        rev_visited[u] = true;
                        rev_stack.push(u);
                        break;
                    }
                }
            }
        }
    }
    rev_visited.iter().all(|&v| v)
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
                    if !reachable[old_v] { continue; }
                    for &v in &unreachable {
                        if v == u || edges[u].contains(&v) { continue; }
                        let cost = match d {
                            0 => candidate_cost(u, v, positions, north_lists),
                            1 => candidate_cost(u, v, positions, south_lists),
                            2 => candidate_cost(u, v, positions, east_lists),
                            _ => candidate_cost(u, v, positions, west_lists),
                        };
                        if cost < f32::MAX {
                            edges[u][d] = v;
                            improved = true;
                            break;
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
                if !can_reach_root[edges[u][0]] && !can_reach_root[edges[u][1]]
                    && !can_reach_root[edges[u][2]] && !can_reach_root[edges[u][3]]
                {
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
                                    edges[u][d] = v;
                                    improved = true;
                                    break;
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

fn compute_reachable(edges: &[[usize; 4]], n: usize, start: usize) -> Vec<bool> {
    let mut visited = vec![false; n];
    let mut stack = vec![start];
    visited[start] = true;
    while let Some(u) = stack.pop() {
        for d in 0..4 {
            let v = edges[u][d];
            if v < n && !visited[v] {
                visited[v] = true;
                stack.push(v);
            }
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
                    can_reach[u] = true;
                    changed = true;
                    break;
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
    torus_dx(xu, xv) + torus_dy(yu, yv)
}
