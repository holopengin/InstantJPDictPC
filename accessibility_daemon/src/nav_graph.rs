use crate::models::BoundingBox;

/// Torus-wrapped horizontal distance.
fn torus_dx(x1: f32, x2: f32) -> f32 {
    let raw = (x1 - x2).abs();
    raw.min(1.0 - raw)
}
/// Torus-wrapped vertical distance.
fn torus_dy(y1: f32, y2: f32) -> f32 {
    let raw = (y1 - y2).abs();
    raw.min(1.0 - raw)
}

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
        const W2: f32 = 1.5; // off‑axis penalty weight (Phase 2 — island connecting)
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

                // Phase 2: SCC-aware fill + connectivity enforcement
        for i in 0..n {
            let mut occupied: [usize; 4] = edges[i];
            let dirs = [(0, &t_north[i]), (1, &t_south[i]), (2, &t_east[i]), (3, &t_west[i])];
            let mut unfilled: Vec<usize> = (0..4).filter(|&d| occupied[d] >= n).collect();
            unfilled.sort_by(|&a, &b| {
                let ca = dirs[a].1.iter().filter(|(v,_)| v != &i && !occupied.contains(v)).count();
                let cb = dirs[b].1.iter().filter(|(v,_)| v != &i && !occupied.contains(v)).count();
                ca.cmp(&cb).then_with(|| a.cmp(&b))
            });
            for &d in &unfilled {
                let list = dirs[d].1;
                let mut chosen = n;
                for &(v, _) in list {
                    if v != i && !occupied.contains(&v) {
                        chosen = v;
                        break;
                    }
                }
                if chosen < n {
                    occupied[d] = chosen;
                    edges[i][d] = chosen;
                }
            }
        }

        // Enforce strong connectivity using Phase 2 candidate lists
        enforce_connectivity(&mut edges, n, &positions, &t_north, &t_south, &t_east, &t_west);

        // Phase 3: wrapping-only candidates (opposite half-plane)
        let mut w3_north: Vec<Vec<(usize, f32)>> = Vec::with_capacity(n);
        let mut w3_south: Vec<Vec<(usize, f32)>> = Vec::with_capacity(n);
        let mut w3_east: Vec<Vec<(usize, f32)>> = Vec::with_capacity(n);
        let mut w3_west: Vec<Vec<(usize, f32)>> = Vec::with_capacity(n);

        for i in 0..n {
            let (xi, yi) = positions[i];
            let mut north: Vec<(usize, f32)> = Vec::new();
            let mut south: Vec<(usize, f32)> = Vec::new();
            let mut east: Vec<(usize, f32)> = Vec::new();
            let mut west: Vec<(usize, f32)> = Vec::new();

            for j in 0..n {
                if i == j { continue; }
                let (xj, yj) = positions[j];
                let tx = torus_dx(xi, xj);
                let ty = torus_dy(yi, yj);
                let xd = (xi - xj).abs();
                let yd = (yi - yj).abs();
                // N wrapped: only nodes physically below
                if yj > yi {
                    let dy_n = (yi - yj + 1.0) % 1.0;
                    if xd <= CONE45 * dy_n { north.push((j, dy_n + W2 * tx)); }
                }
                // S wrapped: only nodes physically above
                if yj < yi {
                    let dy_s = (yj - yi + 1.0) % 1.0;
                    if xd <= CONE45 * dy_s { south.push((j, dy_s + W2 * tx)); }
                }
                // E wrapped: only nodes physically to the left
                if xj < xi {
                    let dx_e = (xj - xi + 1.0) % 1.0;
                    if yd <= CONE45 * dx_e { east.push((j, dx_e + W2 * ty)); }
                }
                // W wrapped: only nodes physically to the right
                if xj > xi {
                    let dx_w = (xi - xj + 1.0) % 1.0;
                    if yd <= CONE45 * dx_w { west.push((j, dx_w + W2 * ty)); }
                }
            }

            let sort_fn = |a: &(usize, f32), b: &(usize, f32)| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.0.cmp(&b.0))
            };
            for list in [&mut north, &mut south, &mut east, &mut west] {
                list.sort_by(sort_fn);
            }
            w3_north.push(north);
            w3_south.push(south);
            w3_east.push(east);
            w3_west.push(west);
        }

        // Phase 3: fill remaining empty slots with wrapping-only candidates
        for i in 0..n {
            let mut occupied: [usize; 4] = edges[i];
            let dirs = [(0, &w3_north[i]), (1, &w3_south[i]), (2, &w3_east[i]), (3, &w3_west[i])];
            let mut unfilled: Vec<usize> = (0..4).filter(|&d| occupied[d] >= n).collect();
            unfilled.sort_by(|&a, &b| {
                let ca = dirs[a].1.iter().filter(|(v,_)| v != &i && !occupied.contains(v)).count();
                let cb = dirs[b].1.iter().filter(|(v,_)| v != &i && !occupied.contains(v)).count();
                ca.cmp(&cb).then_with(|| a.cmp(&b))
            });
            for &d in &unfilled {
                let list = dirs[d].1;
                let mut chosen = n;
                for &(v, _) in list {
                    if v != i && !occupied.contains(&v) {
                        chosen = v;
                        break;
                    }
                }
                if chosen < n {
                    occupied[d] = chosen;
                    edges[i][d] = chosen;
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
    let dir_candidates = [
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

#[cfg(test)]
mod tests {
    //! Dedicated regression tests for [`NavGraph::build`] / [`NavGraph::navigate`].
    //!
    //! Box collection mirrors mobile
    //! `OcrOverlayStateController.rebuildNavGraph` (collect every line's
    //! `charBoxes` in order; build when `boxes.size >= 5`): the PC twin is
    //! `build_nav_graph` (`src/overlay_state.rs:263`). Direction encoding is
    //! `[north, south, east, west] = [0, 1, 2, 3]` on both sides (mobile
    //! `OcrOverlayStateController.navigate` maps DPAD_UP/DOWN/RIGHT/LEFT to
    //! 0/1/2/3; PC `overlay_state.rs:244-250` maps the same). The graph
    //! internals are PC-side per `docs/pipeline/reading-order.md` — the
    //! mobile `nav_graph_core` crate shares the shape (local/greedy/wrap
    //! phases, `(i+1)%n` fallback ring) but not the constants — so these
    //! tests pin exact PC neighbour indices.
    //!
    //! Test names (`nav_graph_0X_...`) are chosen so each can be promoted to
    //! a `nav-graph-0X-...` conformance case later; no corpus cases are added
    //! here.
    use super::*;
    use crate::models::{BoundingBox, RotatedBox};

    /// Direction slots: [north, south, east, west].
    const N: usize = 0;
    const S: usize = 1;
    const E: usize = 2;
    const W: usize = 3;

    fn bb(x: i32, y: i32, w: i32, h: i32) -> BoundingBox {
        BoundingBox::new(x, y, w, h, 1.0)
    }

    fn nav(boxes: &[BoundingBox], idx: usize, dir: usize) -> Option<usize> {
        NavGraph::build(boxes).navigate(idx, dir)
    }

    /// Two horizontal lines of six chars in reading order (the overlay feeds
    /// char boxes line by line, as mobile `rebuildNavGraph` collects
    /// `charBoxes`). 24px chars on a 32px advance, 80px line pitch.
    fn h_lines_2x6() -> Vec<BoundingBox> {
        let mut out = Vec::new();
        for r in 0..2 {
            for c in 0..6 {
                out.push(bb(100 + c * 32, 100 + r * 80, 24, 24));
            }
        }
        out
    }

    /// Two vertical columns of six chars in reading order: the right column
    /// first, since `sort_detected_boxes` orders verticals right-edge-first
    /// (`docs/pipeline/reading-order.md`; mobile `OcrEngine.sortDetectedBoxes`).
    fn v_cols_2x6() -> Vec<BoundingBox> {
        let mut out = Vec::new();
        for &x in &[380, 300] {
            for r in 0..6 {
                out.push(bb(x, 100 + r * 32, 24, 24));
            }
        }
        out
    }

    /// Future conformance case `nav-graph-01-h-lines`: east/west walk along
    /// each line, south drops to the char directly below, north wraps to the
    /// other line (torus). Nothing is within the 0.05 Phase-1 radius at this
    /// spacing, so `initial_edges` stay sentinel and Phase 2+/wrap fill all
    /// slots.
    #[test]
    fn nav_graph_01_horizontal_lines() {
        let boxes = h_lines_2x6();
        let g = NavGraph::build(&boxes);
        assert_eq!(g.n, 12);
        // Phase 1 (strict local, 0.05 radius) finds nothing at this spacing:
        // neighbour pitch 32px over a 284px page extent is 0.11.
        assert!(
            g.initial_edges.iter().all(|e| *e == [12, 12, 12, 12]),
            "no strict-local links at this spacing: {:?}",
            g.initial_edges
        );
        let expect: [[usize; 4]; 12] = [
            [7, 6, 1, 5],
            [6, 7, 2, 0],
            [9, 8, 3, 1],
            [8, 9, 4, 2],
            [9, 10, 5, 3],
            [10, 11, 0, 4],
            [0, 1, 7, 11],
            [1, 0, 8, 6],
            [2, 3, 9, 7],
            [3, 2, 10, 8],
            [4, 3, 11, 9],
            [5, 4, 6, 10],
        ];
        for (i, want) in expect.iter().enumerate() {
            assert_eq!(g.edges[i], *want, "node {i} neighbours");
        }
        // East/west walk the line; line ends wrap around the torus.
        assert_eq!(nav(&boxes, 0, E), Some(1));
        assert_eq!(nav(&boxes, 4, E), Some(5));
        assert_eq!(nav(&boxes, 5, E), Some(0), "line end wraps east");
        assert_eq!(nav(&boxes, 6, W), Some(11), "line start wraps west");
        assert_eq!(nav(&boxes, 7, W), Some(6));
        // South drops to the char directly below; north climbs back.
        assert_eq!(nav(&boxes, 0, S), Some(6));
        assert_eq!(nav(&boxes, 6, N), Some(0));
        assert_eq!(nav(&boxes, 3, S), Some(9));
        assert_eq!(nav(&boxes, 9, N), Some(3));
        // North from the top line wraps to the lower line.
        assert_eq!(nav(&boxes, 1, N), Some(6));
        // South from the bottom line wraps to the upper line.
        assert_eq!(nav(&boxes, 6, S), Some(1));
        // Wrap targets honour the distinct-targets rule, so two top-line
        // nodes can share one wrap target, and the 7-vs-9 wrap-cost tie for
        // node 2 resolves by f32 rounding: both are valid torus wraps.
        assert_eq!(nav(&boxes, 2, N), Some(9));
        assert_eq!(nav(&boxes, 4, N), Some(9));
    }

    /// Future conformance case `nav-graph-02-v-columns`: north/south walk
    /// down each column (column ends wrap), west steps across to the left
    /// column, east from the right column wraps to the left column.
    #[test]
    fn nav_graph_02_vertical_columns() {
        let boxes = v_cols_2x6();
        let g = NavGraph::build(&boxes);
        assert_eq!(g.n, 12);
        assert!(
            g.initial_edges.iter().all(|e| *e == [12, 12, 12, 12]),
            "no strict-local links at this spacing: {:?}",
            g.initial_edges
        );
        let expect: [[usize; 4]; 12] = [
            [5, 1, 7, 6],
            [0, 2, 6, 7],
            [1, 3, 9, 8],
            [2, 4, 8, 9],
            [3, 5, 9, 10],
            [4, 0, 10, 11],
            [11, 7, 0, 1],
            [6, 8, 1, 0],
            [7, 9, 2, 3],
            [8, 10, 3, 2],
            [9, 11, 4, 3],
            [10, 6, 5, 4],
        ];
        for (i, want) in expect.iter().enumerate() {
            assert_eq!(g.edges[i], *want, "node {i} neighbours");
        }
        // South walks down the column; the column foot wraps to its head.
        assert_eq!(nav(&boxes, 0, S), Some(1));
        assert_eq!(nav(&boxes, 4, S), Some(5));
        assert_eq!(nav(&boxes, 5, S), Some(0), "column foot wraps south");
        // North climbs; the column head wraps to its foot (torus).
        assert_eq!(nav(&boxes, 5, N), Some(4));
        assert_eq!(nav(&boxes, 0, N), Some(5), "column head wraps north");
        // West steps across to the left column at the same height.
        assert_eq!(nav(&boxes, 0, W), Some(6));
        assert_eq!(nav(&boxes, 3, W), Some(9));
        // East off the right column wraps to the left column ...
        assert_eq!(nav(&boxes, 6, E), Some(0), "... and back east");
        assert_eq!(nav(&boxes, 1, W), Some(7));
    }

    /// Future conformance case `nav-graph-03-single-line`: one row of six
    /// has no vertical neighbours, so every north/south slot stays the `n`
    /// sentinel and `navigate` returns None; east/west chain with torus wraps
    /// at the ends.
    #[test]
    fn nav_graph_03_single_line_has_no_vertical_neighbours() {
        let boxes: Vec<_> = (0..6).map(|c| bb(100 + c * 32, 100, 24, 24)).collect();
        let g = NavGraph::build(&boxes);
        let expect: [[usize; 4]; 6] = [
            [6, 6, 1, 5],
            [6, 6, 2, 0],
            [6, 6, 3, 1],
            [6, 6, 4, 2],
            [6, 6, 5, 3],
            [6, 6, 0, 4],
        ];
        for (i, want) in expect.iter().enumerate() {
            assert_eq!(g.edges[i], *want, "node {i} neighbours");
        }
        for i in 0..6 {
            assert_eq!(nav(&boxes, i, N), None, "node {i} has no north");
            assert_eq!(nav(&boxes, i, S), None, "node {i} has no south");
        }
        assert_eq!(nav(&boxes, 0, E), Some(1));
        assert_eq!(nav(&boxes, 5, E), Some(0), "row end wraps east");
        assert_eq!(nav(&boxes, 0, W), Some(5), "row start wraps west");
    }

    /// Phase 1 (strict local, 0.05 radius, 45° cone, cost = primary +
    /// 10 × off-axis) links adjacent chars directly once the neighbour pitch
    /// drops under 0.05 of the page extent: 25 chars on a 32px advance span
    /// 792px, so the pitch is 0.040. `initial_edges` already hold the
    /// east/west neighbours; north/south stay empty on a single row.
    #[test]
    fn nav_graph_03b_long_row_links_phase1_local() {
        let boxes: Vec<_> = (0..25).map(|c| bb(100 + c * 32, 100, 24, 24)).collect();
        let g = NavGraph::build(&boxes);
        assert_eq!(g.n, 25);
        // Mid-row node: Phase 1 already picked the adjacent neighbours.
        assert_eq!(g.initial_edges[12], [25, 25, 13, 11]);
        assert_eq!(g.initial_edges[1], [25, 25, 2, 0]);
        // Row ends have no local wrap partner: west/east fill in Phase 3.
        assert_eq!(g.initial_edges[0][W], 25);
        assert_eq!(g.initial_edges[24][E], 25);
        // Final graph: a clean east/west ring, no vertical neighbours.
        for i in 0..25 {
            assert_eq!(g.edges[i][E], (i + 1) % 25, "node {i} east");
            assert_eq!(g.edges[i][W], (i + 24) % 25, "node {i} west");
            assert_eq!(g.edges[i][N], 25, "node {i} has no north");
            assert_eq!(g.edges[i][S], 25, "node {i} has no south");
        }
        assert_eq!(nav(&boxes, 24, E), Some(0));
        assert_eq!(nav(&boxes, 0, W), Some(24));
        assert_eq!(nav(&boxes, 12, N), None);
    }

    /// Future conformance case `nav-graph-04-fallback`: fewer than five nodes
    /// take the `(i+1)%n` ring (`NavGraph::fallback`), the same formula as
    /// mobile `nav_graph_core ... fallback`. Empty input builds an empty
    /// graph where every `navigate` returns None.
    #[test]
    fn nav_graph_04_small_inputs_take_the_fallback_ring() {
        // Empty: no nodes, no edges, no navigation.
        let g = NavGraph::build(&[]);
        assert_eq!(g.n, 0);
        assert!(g.edges.is_empty());
        assert_eq!(g.navigate(0, N), None);
        // Single char: the ring is all self-loops, so navigation stays put.
        // (Mobile `fallback` computes the identical `(i+1)%1 = 0` entries.)
        let one = [bb(100, 100, 24, 24)];
        let g = NavGraph::build(&one);
        assert_eq!(g.edges, vec![[0, 0, 0, 0]]);
        for dir in [N, S, E, W] {
            assert_eq!(g.navigate(0, dir), Some(0), "single char stays put");
        }
        // Two nodes: [(i+1)%2, (i+2)%2, (i+3)%2, (i+4)%2].
        let two: Vec<_> = (0..2).map(|c| bb(100 + c * 32, 100, 24, 24)).collect();
        let g = NavGraph::build(&two);
        assert_eq!(g.edges, vec![[1, 0, 1, 0], [0, 1, 0, 1]]);
        assert_eq!(g.navigate(0, E), Some(1));
        assert_eq!(g.navigate(1, W), Some(1), "two-node ring west is self");
        // Four nodes: the full ring.
        let four: Vec<_> = (0..4).map(|c| bb(100 + c * 32, 100, 24, 24)).collect();
        let g = NavGraph::build(&four);
        assert_eq!(
            g.edges,
            vec![[1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2], [0, 1, 2, 3]]
        );
        assert_eq!(g.navigate(3, N), Some(0), "four-node ring wraps");
    }

    /// Future conformance case `nav-graph-05-mixed`: three horizontal chars
    /// then three vertical chars (reading order is horizontals-first per
    /// `docs/pipeline/reading-order.md`). East reaches from the line end into
    /// the column head; the column's west reaches back; the 45° cone leaves
    /// genuinely empty directions (`n` sentinel → `navigate` None).
    #[test]
    fn nav_graph_05_mixed_orientation_page() {
        let mut boxes = Vec::new();
        for c in 0..3 {
            boxes.push(bb(100 + c * 32, 100, 24, 24));
        }
        for r in 0..3 {
            boxes.push(bb(400, 100 + r * 32, 24, 24));
        }
        let g = NavGraph::build(&boxes);
        assert_eq!(g.n, 6);
        let expect: [[usize; 4]; 6] = [
            [4, 6, 1, 3],
            [5, 6, 2, 0],
            [5, 6, 3, 1],
            [5, 4, 0, 2],
            [3, 5, 0, 2],
            [4, 3, 1, 2],
        ];
        for (i, want) in expect.iter().enumerate() {
            assert_eq!(g.edges[i], *want, "node {i} neighbours");
        }
        // The line end reaches east into the column head ...
        assert_eq!(nav(&boxes, 2, E), Some(3));
        // ... and the column head reaches back west along the line.
        assert_eq!(nav(&boxes, 3, W), Some(2));
        // The column walks north/south ...
        assert_eq!(nav(&boxes, 3, S), Some(4));
        assert_eq!(nav(&boxes, 5, N), Some(4));
        assert_eq!(nav(&boxes, 4, S), Some(5));
        // ... while the line's south looks past the far-right column
        // (outside the 45° cone), so it stays empty.
        assert_eq!(nav(&boxes, 0, S), None);
        assert_eq!(nav(&boxes, 2, S), None);
        // Column mid reaches east back to the line start (wrap).
        assert_eq!(nav(&boxes, 4, E), Some(0));
    }

    /// The near-square-horizontal rule (`RotatedBox::is_vertical`, the exact
    /// predicate `sort_detected_boxes` uses at `src/ocr_engine.rs:1545` to
    /// split horizontals from verticals like mobile `isVerticalLineBox`):
    /// a frame counts as vertical only at 1.25× elongation, so a lone
    /// upright character is never fed sideways.
    #[test]
    fn nav_graph_06a_near_square_counts_as_horizontal() {
        let frame = |w: f32, h: f32| RotatedBox::new(0.0, 0.0, w, h, 0.0, 1.0);
        assert!(!frame(40.0, 40.0).is_vertical(), "square is horizontal");
        assert!(!frame(40.0, 44.0).is_vertical(), "near-square is horizontal");
        assert!(frame(32.0, 40.0).is_vertical(), "1.25x boundary is vertical");
        assert!(!frame(33.0, 40.0).is_vertical(), "below 1.25x is horizontal");
        assert!(frame(30.0, 200.0).is_vertical(), "column is vertical");
        assert!(!frame(200.0, 30.0).is_vertical(), "line is horizontal");
    }

    /// Future conformance case `nav-graph-06-near-square`: a 40×40 box inline
    /// in a horizontal line (same centres as a 24px char would have) sorts
    /// with the horizontals and navigates east/west along the line — it is
    /// never treated as a vertical column.
    #[test]
    fn nav_graph_06b_near_square_box_navigates_with_its_line() {
        let mut boxes = Vec::new();
        for c in 0..6 {
            if c == 2 {
                boxes.push(bb(100 + c * 32 - 8, 92, 40, 40));
            } else {
                boxes.push(bb(100 + c * 32, 100, 24, 24));
            }
        }
        // The near-square frame classifies horizontal by the real rule ...
        let quad = RotatedBox::new(176.0, 112.0, 40.0, 40.0, 0.0, 1.0);
        assert!(!quad.is_vertical());
        // ... and the graph walks straight through it along the line.
        let g = NavGraph::build(&boxes);
        let expect: [[usize; 4]; 6] = [
            [6, 6, 1, 5],
            [6, 6, 2, 0],
            [6, 6, 3, 1],
            [6, 6, 4, 2],
            [6, 6, 5, 3],
            [6, 6, 0, 4],
        ];
        for (i, want) in expect.iter().enumerate() {
            assert_eq!(g.edges[i], *want, "node {i} neighbours");
        }
        assert_eq!(nav(&boxes, 1, E), Some(2), "east into the near-square box");
        assert_eq!(nav(&boxes, 3, W), Some(2), "west into the near-square box");
        assert_eq!(nav(&boxes, 2, E), Some(3), "east out of the near-square box");
        assert_eq!(nav(&boxes, 2, W), Some(1), "west out of the near-square box");
    }

    /// Future conformance case `nav-graph-07-corpus-mixed`: the
    /// `reading-order-01-mixed` corpus geometry in reading order
    /// (horizontals including the near-square box first, then verticals
    /// right-edge-first). The second line reaches south into the near-square
    /// box; the left vertical reaches east across to the right column while
    /// the right column's west reaches back; exhausted directions stay empty.
    #[test]
    fn nav_graph_07_corpus_mixed_geometry_in_reading_order() {
        // Corpus boxes reordered to `expect_order` [0, 1, 4, 3, 2].
        let boxes = [
            bb(10, 10, 200, 30),
            bb(10, 100, 200, 30),
            bb(10, 200, 40, 40),
            bb(300, 10, 30, 200),
            bb(100, 10, 30, 200),
        ];
        // The corpus order itself follows the real rule: the 40×40 frame is
        // horizontal, the 30×200 frames vertical.
        assert!(!RotatedBox::new(0.0, 0.0, 40.0, 40.0, 0.0, 1.0).is_vertical());
        assert!(RotatedBox::new(0.0, 0.0, 30.0, 200.0, 0.0, 1.0).is_vertical());
        let g = NavGraph::build(&boxes);
        assert_eq!(g.n, 5);
        let expect: [[usize; 4]; 5] = [
            [4, 1, 3, 5],
            [4, 2, 3, 5],
            [1, 4, 3, 5],
            [1, 0, 5, 4],
            [0, 1, 3, 5],
        ];
        for (i, want) in expect.iter().enumerate() {
            assert_eq!(g.edges[i], *want, "node {i} neighbours");
        }
        // Down the horizontals into the near-square box ...
        assert_eq!(nav(&boxes, 0, S), Some(1));
        assert_eq!(nav(&boxes, 1, S), Some(2));
        assert_eq!(nav(&boxes, 2, N), Some(1));
        // ... east into the right vertical, west back across the columns.
        assert_eq!(nav(&boxes, 0, E), Some(3));
        assert_eq!(nav(&boxes, 4, E), Some(3));
        assert_eq!(nav(&boxes, 3, W), Some(4));
        // West off the horizontals is exhausted (both wrap candidates are
        // already taken by other directions), so navigation stops.
        assert_eq!(nav(&boxes, 0, W), None);
        assert_eq!(nav(&boxes, 3, E), None);
    }

    /// `navigate` resolves sentinel slots and out-of-range inputs to None
    /// (mobile `nav_graph_core ... navigate` returns None the same way).
    #[test]
    fn nav_graph_08_navigate_bounds() {
        let boxes: Vec<_> = (0..6).map(|c| bb(100 + c * 32, 100, 24, 24)).collect();
        let g = NavGraph::build(&boxes);
        // Sentinel slot (single row has no north).
        assert_eq!(g.navigate(0, N), None);
        // Out-of-range node and direction.
        assert_eq!(g.navigate(6, N), None);
        assert_eq!(g.navigate(99, E), None);
        assert_eq!(g.navigate(0, 4), None);
        assert_eq!(g.navigate(0, 99), None);
        // Empty graph: everything is out of range.
        let empty = NavGraph::build(&[]);
        assert_eq!(empty.navigate(0, N), None);
    }
}
