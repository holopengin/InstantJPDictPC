use crate::models::BoundingBox;
use crate::nav_graph_index::NavIndex;

/// Torus-wrapped horizontal distance (used by the differential oracle).
#[cfg(test)]
fn torus_dx(x1: f32, x2: f32) -> f32 {
    let raw = (x1 - x2).abs();
    raw.min(1.0 - raw)
}
/// Torus-wrapped vertical distance (used by the differential oracle).
#[cfg(test)]
fn torus_dy(y1: f32, y2: f32) -> f32 {
    let raw = (y1 - y2).abs();
    raw.min(1.0 - raw)
}

/// Number of candidates any single consumer can need from one sorted list.
/// At the moment a walker scans a list it holds at most four distinct
/// used/occupied targets, so among any five distinct candidates one is free;
/// when the list is shorter a five-candidate read *is* the whole list, so an
/// empty answer means the list is genuinely exhausted (which reproduces the
/// old `for j in 0..n` fallback without scanning).
const TOP_K: usize = 5;

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

        // One k-d tree answers every phase's candidate enumeration and count.
        // Each phase keeps its own predicate and cost (see nav_graph_index),
        // so the output is byte-identical to the old materialised-list build.
        let index = NavIndex::new(&positions);

        // Phase 1: greedy local assignment.
        let mut edges = vec![[n; 4]; n];
        for i in 0..n {
            edges[i] = greedy_assignment_indexed(i, n, &index);
        }
        let initial_edges = edges.clone();

        // Phase 2: SCC-aware fill + connectivity enforcement.
        fill_phase(&mut edges, n, &index, 2);
        enforce_connectivity(&mut edges, n, &positions);

        // Phase 3: wrap fill.
        fill_phase(&mut edges, n, &index, 3);

        Self { edges, initial_edges, positions, n }
    }

    /// The pre-#107 implementation, kept verbatim as a differential oracle
    /// for the enumerator/index tests. Materialises the sorted candidate
    /// lists; never compiled outside `cfg(test)`.
    #[cfg(test)]
    pub fn build_oracle(boxes: &[BoundingBox]) -> Self {
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
        enforce_connectivity_oracle(&mut edges, n, &positions, &t_north, &t_south, &t_east, &t_west);

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

/// Oracle: the pre-#107 greedy assignment over materialised lists.
#[cfg(test)]
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

/// Oracle: the pre-#107 greedy assignment for one node.
#[cfg(test)]
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

/// Phase 1 greedy assignment against the exact enumerator. Mirrors
/// [greedy_assignment] step for step; the only change is that a direction's
/// "first candidate not already used" is the top of an exact top-[TOP_K]
/// query excluding the used set. Five candidates always suffice: at the scan
/// point the used set holds at most four distinct targets, so one of the five
/// is free, and a shorter list is fully enumerated.
fn greedy_assignment_indexed(i: usize, n: usize, index: &NavIndex) -> [usize; 4] {
    let mut result = [n; 4];
    for d in 0..4 {
        result[d] = index
            .top_k(i, 1, d as u8, &[], TOP_K)
            .first()
            .copied()
            .unwrap_or(n);
    }

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

        used.clear();
        for d in 0..4 {
            if result[d] == n || !used.insert(result[d]) {
                let original = result[d];
                let used_vec: Vec<usize> = used.iter().copied().collect();
                let alt = index
                    .top_k(i, 1, d as u8, &used_vec, TOP_K)
                    .first()
                    .copied();
                if let Some(alt) = alt {
                    result[d] = alt;
                    used.insert(alt);
                    has_conflict = true;
                } else if original != n {
                    // The list had no free candidate: reproduce the old
                    // "for j in 0..n" fallback verbatim.
                    for j in 0..n {
                        if !used.contains(&j) {
                            result[d] = j;
                            used.insert(j);
                            has_conflict = true;
                            break;
                        }
                    }
                }
                break; // fix one conflict per iteration
            }
        }
    }

    result
}

/// Phase 2 / Phase 3 fill against the exact enumerator. Mirrors the old fill:
/// unfilled directions are ordered by how many candidates remain (fewest
/// first, then direction), and each is filled with its first free candidate.
fn fill_phase(edges: &mut [[usize; 4]], n: usize, index: &NavIndex, phase: u8) {
    for i in 0..n {
        let occupied0 = edges[i];

        // Unique real targets already occupied (sentinel n is not a target).
        let mut occ_set: Vec<usize> = Vec::new();
        for &v in &occupied0 {
            if v < n && !occ_set.contains(&v) {
                occ_set.push(v);
            }
        }

        let mut unfilled: Vec<usize> = (0..4).filter(|&d| occupied0[d] >= n).collect();

        // Exact full-list counts, minus the occupied targets that appear in
        // the list (the old ".filter(!occupied.contains(v)).count()").
        let mut counts = [0usize; 4];
        for &d in &unfilled {
            let mut c = index.count(i, phase, d as u8);
            let (xi, yi) = index.position(i);
            for &v in &occ_set {
                let (xj, yj) = index.position(v);
                let xd = (xi - xj).abs();
                let yd = (yi - yj).abs();
                let euc = (xd * xd + yd * yd).sqrt();
                if crate::nav_graph_index::in_region(phase, d as u8, xi, yi, xj, yj, euc) {
                    c = c.saturating_sub(1);
                }
            }
            counts[d] = c;
        }

        unfilled.sort_by(|&a, &b| counts[a].cmp(&counts[b]).then_with(|| a.cmp(&b)));

        let mut occupied = occupied0;
        for &d in &unfilled {
            let excl: Vec<usize> = occupied.iter().copied().filter(|&v| v < n).collect();
            if let Some(&v) = index.top_k(i, phase, d as u8, &excl, TOP_K).first() {
                occupied[d] = v;
                edges[i][d] = v;
            }
        }
    }
}

/// Oracle: the pre-#107 per-phase candidate-list construction, over the raw
/// normalised positions. Used only to prove the k-d tree enumerators exact.
#[cfg(test)]
fn build_lists_oracle(
    positions: &[(f32, f32)],
    phase: u8,
) -> Vec<[Vec<(usize, f32)>; 4]> {
    let n = positions.len();
    const W: f32 = 10.0;
    const W2: f32 = 1.5;
    const LOCAL_DIST: f32 = 0.05;
    const CONE45: f32 = 1.0;

    let mut out: Vec<[Vec<(usize, f32)>; 4]> = Vec::with_capacity(n);
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
            match phase {
                1 => {
                    let euc = (xd * xd + yd * yd).sqrt();
                    if euc > LOCAL_DIST { continue; }
                    let dy_n = yi - yj;
                    if yj < yi && xd <= CONE45 * dy_n { north.push((j, dy_n + W * xd)); }
                    let dy_s = yj - yi;
                    if yj > yi && xd <= CONE45 * dy_s { south.push((j, dy_s + W * xd)); }
                    let dx_e = xj - xi;
                    if xj > xi && yd <= CONE45 * dx_e { east.push((j, dx_e + W * yd)); }
                    let dx_w = xi - xj;
                    if xj < xi && yd <= CONE45 * dx_w { west.push((j, dx_w + W * yd)); }
                }
                2 => {
                    let dy_n = yi - yj;
                    if yj < yi && xd <= CONE45 * dy_n { north.push((j, dy_n + W2 * xd)); }
                    let dy_s = yj - yi;
                    if yj > yi && xd <= CONE45 * dy_s { south.push((j, dy_s + W2 * xd)); }
                    let dx_e = xj - xi;
                    if xj > xi && yd <= CONE45 * dx_e { east.push((j, dx_e + W2 * yd)); }
                    let dx_w = xi - xj;
                    if xj < xi && yd <= CONE45 * dx_w { west.push((j, dx_w + W2 * yd)); }
                }
                _ => {
                    let tx = torus_dx(xi, xj);
                    let ty = torus_dy(yi, yj);
                    if yj > yi {
                        let dy_n = (yi - yj + 1.0) % 1.0;
                        if xd <= CONE45 * dy_n { north.push((j, dy_n + W2 * tx)); }
                    }
                    if yj < yi {
                        let dy_s = (yj - yi + 1.0) % 1.0;
                        if xd <= CONE45 * dy_s { south.push((j, dy_s + W2 * tx)); }
                    }
                    if xj < xi {
                        let dx_e = (xj - xi + 1.0) % 1.0;
                        if yd <= CONE45 * dx_e { east.push((j, dx_e + W2 * ty)); }
                    }
                    if xj > xi {
                        let dx_w = (xi - xj + 1.0) % 1.0;
                        if yd <= CONE45 * dx_w { west.push((j, dx_w + W2 * ty)); }
                    }
                }
            }
        }

        match phase {
            1 => {
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
            }
            _ => {
                let sort_fn = |a: &(usize, f32), b: &(usize, f32)| {
                    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.0.cmp(&b.0))
                };
                for list in [&mut north, &mut south, &mut east, &mut west] {
                    list.sort_by(sort_fn);
                }
            }
        }

        out.push([north, south, east, west]);
    }
    out
}

fn reverse_adjacency(edges: &[[usize; 4]], n: usize) -> Vec<Vec<usize>> {
    let mut radj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for u in 0..n {
        for d in 0..4 {
            let v = edges[u][d];
            if v < n {
                // At most four outgoing edges per node; a duplicate target in
                // two directions is harmless for reachability.
                radj[v].push(u);
            }
        }
    }
    radj
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

    // Reverse reachability from 0 via reverse adjacency — O(n + E), and the
    // reachable set is order-independent, so this is identical to the old
    // O(n²) scan.
    let radj = reverse_adjacency(edges, n);
    let mut rev_visited = vec![false; n];
    let mut rev_stack = vec![0usize];
    rev_visited[0] = true;
    while let Some(v) = rev_stack.pop() {
        for &u in &radj[v] {
            if !rev_visited[u] { rev_visited[u] = true; rev_stack.push(u); }
        }
    }
    rev_visited.iter().all(|&v| v)
}

/// Oracle-only: forward reachability, kept for the pre-#107 enforcement.
#[cfg(test)]
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

/// Oracle-only: reverse reachability via reverse adjacency.
#[cfg(test)]
fn compute_reverse_reachable(edges: &[[usize; 4]], n: usize, target: usize) -> Vec<bool> {
    let radj = reverse_adjacency(edges, n);
    let mut can_reach = vec![false; n];
    let mut stack = vec![target];
    can_reach[target] = true;
    while let Some(v) = stack.pop() {
        for &u in &radj[v] {
            if !can_reach[u] { can_reach[u] = true; stack.push(u); }
        }
    }
    can_reach
}

/// Oracle: the pre-#107 candidate_cost. Its return value never mattered -
/// [enforce_connectivity_oracle] only tests "cost < f32::MAX", which the L1
/// fallback always satisfies - so the non-oracle rewrite drops it.
#[cfg(test)]
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

/// Oracle: the pre-#107 connectivity enforcement, kept verbatim.
#[cfg(test)]
fn enforce_connectivity_oracle(
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

/// Signed shortest torus offset from `a` to `b` (in `[-0.5, 0.5]`), for
/// picking a sensible direction for a repair edge.
fn torus_off(a: f32, b: f32) -> f32 {
    let mut d = b - a;
    if d > 0.5 {
        d -= 1.0;
    } else if d < -0.5 {
        d += 1.0;
    }
    d
}

/// The cardinal direction whose cone best matches `to` relative to `from`
/// (dominant torus axis decides).
fn suggested_dir(from: usize, to: usize, positions: &[(f32, f32)]) -> usize {
    let (xf, yf) = positions[from];
    let (xt, yt) = positions[to];
    let dx = torus_off(xf, xt);
    let dy = torus_off(yf, yt);
    if dy.abs() >= dx.abs() {
        if dy < 0.0 { 0 } else { 1 } // north / south
    } else if dx > 0.0 {
        2 // east
    } else {
        3 // west
    }
}

/// Place a repair edge `from -> to`. Prefer an empty slot, best-aligned
/// direction first; only if every slot is occupied, overwrite the slot whose
/// current target is farthest (the least "reasonable navigation" edge).
/// Returns `true` when an empty slot was used (no existing edge disturbed).
fn place_edge(
    edges: &mut [[usize; 4]],
    from: usize,
    to: usize,
    n: usize,
    positions: &[(f32, f32)],
) -> bool {
    let best = suggested_dir(from, to, positions);
    if edges[from][best] >= n {
        edges[from][best] = to;
        return true;
    }
    for d in 0..4 {
        if edges[from][d] >= n {
            edges[from][d] = to;
            return true;
        }
    }
    // Every slot is occupied: overwrite the farthest target, deterministically.
    let (xf, yf) = positions[from];
    let mut worst = 0usize;
    let mut worst_d = f32::NEG_INFINITY;
    for d in 0..4 {
        let v = edges[from][d];
        if v >= n {
            continue;
        }
        let (xv, yv) = positions[v];
        let dx = torus_off(xf, xv);
        let dy = torus_off(yf, yv);
        let dist = (dx * dx + dy * dy).sqrt();
        if dist > worst_d {
            worst_d = dist;
            worst = d;
        }
    }
    edges[from][worst] = to;
    false
}

/// Deterministic strongly-connected-components labelling via Kosaraju
/// (iterative, so deep graphs cannot overflow the stack).
fn scc_of(edges: &[[usize; 4]], radj: &[Vec<usize>], n: usize) -> Vec<usize> {
    let mut visited = vec![false; n];
    let mut order: Vec<usize> = Vec::with_capacity(n);
    for s in 0..n {
        if visited[s] {
            continue;
        }
        visited[s] = true;
        let mut stack: Vec<(usize, u8)> = vec![(s, 0)];
        while let Some(&(u, d)) = stack.last() {
            if (d as usize) < 4 {
                stack.last_mut().unwrap().1 += 1;
                let v = edges[u][d as usize];
                if v < n && !visited[v] {
                    visited[v] = true;
                    stack.push((v, 0));
                }
            } else {
                order.push(u);
                stack.pop();
            }
        }
    }

    let mut comp = vec![usize::MAX; n];
    let mut num = 0usize;
    for &s in order.iter().rev() {
        if comp[s] != usize::MAX {
            continue;
        }
        comp[s] = num;
        let mut stack = vec![s];
        while let Some(u) = stack.pop() {
            for &w in &radj[u] {
                if comp[w] == usize::MAX {
                    comp[w] = num;
                    stack.push(w);
                }
            }
        }
        num += 1;
    }
    comp
}

/// Connectivity enforcement: make the graph strongly connected with minimal
/// disruption, in one provably-terminating O(n + E) pass.
///
/// The old implementation was a swap loop that ran at most `n` iterations,
/// performed one swap each and — on row, column and scatter layouts — never
/// reached strong connectivity at all (a 1000-node row ended with 899
/// in-degree-0 nodes and had overwritten good navigation edges). It also never
/// filled an empty slot, so isolated nodes could not be connected.
///
/// The repair works on the SCC condensation DAG. It closes a directed cycle
/// through every component: for each component `a` lacking a direct edge to
/// the next component `b` it adds `rep(a) -> rep(b)`. The literal cycle
/// `c_0 -> c_1 -> ... -> c_{m-1} -> c_0` then exists, so every component (and
/// every node) reaches every other. Each component contributes at most one
/// added edge, written last for its source node, so a later iteration never
/// removes it — the construction is correct even when a component has no empty
/// slot and `place_edge` must overwrite one of its own outgoing edges.
///
/// Added edges go into empty slots wherever possible, so the
/// reasonable-navigation edges produced by Phases 1–2 survive; the
/// best-aligned direction is chosen, and a slot is only overwritten when the
/// component has no empty slot anywhere.
fn enforce_connectivity(edges: &mut [[usize; 4]], n: usize, positions: &[(f32, f32)]) {
    if n == 0 || is_strongly_connected(edges, n) {
        return;
    }
    let radj = reverse_adjacency(edges, n);
    let comp = scc_of(edges, &radj, n);
    let num = comp.iter().copied().max().map(|m| m + 1).unwrap_or(0);
    if num <= 1 {
        return;
    }

    // Condensation adjacency; dedup not required for the membership test.
    let mut cadj: Vec<Vec<usize>> = vec![Vec::new(); num];
    for u in 0..n {
        for d in 0..4 {
            let v = edges[u][d];
            if v < n && comp[u] != comp[v] {
                cadj[comp[u]].push(comp[v]);
            }
        }
    }

    // Smallest-index node per SCC, preferring one with an empty slot so a
    // repair edge is placed without disturbing navigation.
    let mut rep = vec![usize::MAX; num];
    let mut rep_empty = vec![usize::MAX; num];
    for u in 0..n {
        let c = comp[u];
        if rep[c] == usize::MAX {
            rep[c] = u;
        }
        if rep_empty[c] == usize::MAX && edges[u].iter().any(|&v| v >= n) {
            rep_empty[c] = u;
        }
    }
    for c in 0..num {
        if rep_empty[c] == usize::MAX {
            rep_empty[c] = rep[c];
        }
    }

    // Close a directed cycle through every component.
    for a in 0..num {
        let b = (a + 1) % num;
        if a == b || cadj[a].contains(&b) {
            continue;
        }
        place_edge(edges, rep_empty[a], rep[b], n, positions);
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

    // ------------------------------------------------------------------
    // Issue #107 differential oracle: the k-d tree enumerators must return
    // exactly the prefixes of the old materialised lists, and the whole build
    // must reproduce `build_oracle` byte for byte.
    // ------------------------------------------------------------------

    /// Deterministic LCG so the scatter layouts are reproducible.
    struct Lcg(u64);
    impl Lcg {
        fn new(seed: u64) -> Self { Lcg(seed.wrapping_mul(6364136223846793005).wrapping_add(1)) }
        fn next_f(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (((self.0 >> 33) as u32) as f32) / (u32::MAX as f32)
        }
    }

    fn norm_positions(boxes: &[BoundingBox]) -> Vec<(f32, f32)> {
        let mx = boxes.iter().map(|b| b.right() as f32).fold(0f32, f32::max).max(1.0);
        let my = boxes.iter().map(|b| b.bottom() as f32).fold(0f32, f32::max).max(1.0);
        boxes.iter().map(|b| {
            ((b.left() as f32 + b.w as f32 / 2.0) / mx, (b.top() as f32 + b.h as f32 / 2.0) / my)
        }).collect()
    }

    /// Canonical layouts plus seeded random scatter and a few adversarial ones.
    fn oracle_layouts() -> Vec<(&'static str, Vec<BoundingBox>)> {
        let mut out: Vec<(&'static str, Vec<BoundingBox>)> = Vec::new();
        out.push(("h_lines_2x6", h_lines_2x6()));
        out.push(("v_cols_2x6", v_cols_2x6()));
        out.push(("single_row", (0..6).map(|c| bb(100 + c * 32, 100, 24, 24)).collect()));
        out.push(("long_row", (0..25).map(|c| bb(100 + c * 32, 100, 24, 24)).collect()));
        out.push(("mixed", {
            let mut v = Vec::new();
            for c in 0..3 { v.push(bb(100 + c * 32, 100, 24, 24)); }
            for r in 0..3 { v.push(bb(400, 100 + r * 32, 24, 24)); }
            v
        }));
        out.push(("corpus", vec![
            bb(10, 10, 200, 30), bb(10, 100, 200, 30), bb(10, 200, 40, 40),
            bb(300, 10, 30, 200), bb(100, 10, 30, 200),
        ]));
        out.push(("grid_7x7", (0..49).map(|k| bb(50 + (k % 7) * 30, 50 + (k / 7) * 30, 24, 24)).collect()));
        out.push(("row_100", (0..100).map(|c| bb(10 + c * 8, 100, 6, 6)).collect()));
        out.push(("col_100", (0..100).map(|r| bb(100, 10 + r * 8, 6, 6)).collect()));
        // Coincident points: every Phase-1 list is O(n).
        out.push(("dupes_40", (0..40).map(|_| bb(100, 100, 24, 24)).collect()));
        // Tight clusters (the layout the old top-1 truncation got wrong).
        out.push(("clusters", {
            let mut v = Vec::new();
            for c in 0..8 {
                for k in 0..6 {
                    v.push(bb(100 + c * 400 + k, 100 + (k % 3) * 30, 24, 24));
                }
            }
            v
        }));
        // Single diagonal.
        out.push(("diag", (0..30).map(|k| bb(50 + k * 40, 50 + k * 40, 20, 20)).collect()));
        // Seeded random scatters.
        for &(seed, n) in &[(7u64, 40usize), (11, 80), (23, 200), (101, 512), (555, 137)] {
            let mut rng = Lcg::new(seed);
            let v: Vec<_> = (0..n).map(|_| {
                bb((rng.next_f() * 1800.0) as i32, (rng.next_f() * 1200.0) as i32, 24, 24)
            }).collect();
            out.push(("scatter", v));
        }
        // Random scatter with many duplicates: forces O(n) Phase-1 lists.
        out.push(("scatter_dupes", {
            let mut rng = Lcg::new(31337);
            (0..120).map(|_| {
                let x = 100 + ((rng.next_f() * 4.0) as i32) * 10;
                let y = 100 + ((rng.next_f() * 4.0) as i32) * 10;
                bb(x, y, 24, 24)
            }).collect()
        }));
        // Points hugging the page edge (torus wrap stress).
        out.push(("near_edge", {
            let mut rng = Lcg::new(99);
            (0..60).map(|k| {
                if k % 2 == 0 {
                    bb((rng.next_f() * 30.0) as i32, (rng.next_f() * 1200.0) as i32, 24, 24)
                } else {
                    bb(1770 + (rng.next_f() * 30.0) as i32, (rng.next_f() * 1200.0) as i32, 24, 24)
                }
            }).collect()
        }));
        out
    }

    /// The enumerator must return exactly the old list's first `TOP_K`, and
    /// the count must equal the old list length, for every phase/direction and
    /// every layout.
    #[test]
    fn nav_graph_09_enumerators_match_the_oracle_lists() {
        for (name, boxes) in oracle_layouts() {
            let pos = norm_positions(&boxes);
            let index = NavIndex::new(&pos);
            for phase in [1u8, 2, 3] {
                let lists = build_lists_oracle(&pos, phase);
                for i in 0..pos.len() {
                    for dir in 0..4usize {
                        let full = &lists[i][dir];
                        let want_prefix: Vec<usize> =
                            full.iter().take(TOP_K).map(|(v, _)| *v).collect();
                        let got = index.top_k(i, phase, dir as u8, &[], TOP_K);
                        assert_eq!(
                            got, want_prefix,
                            "{name} phase {phase} node {i} dir {dir}: top-{TOP_K} mismatch"
                        );
                        if phase != 1 {
                            assert_eq!(
                                index.count(i, phase, dir as u8),
                                full.len(),
                                "{name} phase {phase} node {i} dir {dir}: count mismatch"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Exclusion must behave like "skip already-used candidates", matching the
    /// old walkers' first-free-pick exactly.
    #[test]
    fn nav_graph_10_enumerators_honour_exclusions() {
        let mut rng = Lcg::new(4242);
        for (name, boxes) in oracle_layouts() {
            let pos = norm_positions(&boxes);
            let index = NavIndex::new(&pos);
            for phase in [1u8, 2, 3] {
                let lists = build_lists_oracle(&pos, phase);
                for i in 0..pos.len() {
                    for dir in 0..4usize {
                        let full: Vec<usize> = lists[i][dir].iter().map(|(v, _)| *v).collect();
                        // A few exclusion shapes: prefix, suffix, and random.
                        let shapes: Vec<Vec<usize>> = vec![
                            full.iter().take(2).copied().collect(),
                            full.iter().rev().take(3).copied().collect(),
                            {
                                let mut e = Vec::new();
                                for _ in 0..4 {
                                    if !full.is_empty() {
                                        let k = (rng.next_f() * full.len() as f32) as usize;
                                        e.push(full[k.min(full.len() - 1)]);
                                    }
                                }
                                e
                            },
                        ];
                        for excl in shapes {
                            let want: Vec<usize> = full.iter()
                                .filter(|v| !excl.contains(v))
                                .take(TOP_K)
                                .copied()
                                .collect();
                            let got = index.top_k(i, phase, dir as u8, &excl, TOP_K);
                            assert_eq!(
                                got, want,
                                "{name} phase {phase} node {i} dir {dir}: exclusion {excl:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// After the whole build the graph must be strongly connected on every
    /// layout — the property the old enforcement failed to establish on row,
    /// column and scatter pages.
    #[test]
    fn nav_graph_11_build_is_strongly_connected() {
        for (name, boxes) in oracle_layouts() {
            let g = NavGraph::build(&boxes);
            assert!(
                is_strongly_connected(&g.edges, g.n),
                "{name} (n={}): not strongly connected after build",
                g.n
            );
        }
    }

    /// The enforcement fix must not disturb a graph the old code had already
    /// made strongly connected without overwriting navigation: every layout
    /// whose old graph equals its pre-enforcement edges stays byte identical,
    /// and every layout that changes goes from not-strongly-connected (or an
    /// over-eager swap repair) to strongly connected, without sacrificing any
    /// more Phase-1 (`initial_edges`) links than the old repair did.
    #[test]
    fn nav_graph_12_enforcement_preserves_working_graphs() {
        let initial_preserved = |g: &NavGraph| -> usize {
            let mut c = 0;
            for i in 0..g.n {
                for d in 0..4 {
                    if g.initial_edges[i][d] < g.n && g.edges[i][d] == g.initial_edges[i][d] {
                        c += 1;
                    }
                }
            }
            c
        };
        for (name, boxes) in oracle_layouts() {
            let fast = NavGraph::build(&boxes);
            let slow = NavGraph::build_oracle(&boxes);
            assert_eq!(fast.n, slow.n, "{name}: n");
            assert_eq!(fast.initial_edges, slow.initial_edges, "{name}: initial_edges");
            assert!(
                is_strongly_connected(&fast.edges, fast.n),
                "{name}: new graph is not strongly connected"
            );
            if fast.edges == slow.edges {
                continue; // byte-identical: the fix changed nothing here
            }
            assert!(
                initial_preserved(&fast) >= initial_preserved(&slow),
                "{name}: new repair disturbed more Phase-1 links than the old one"
            );
        }
    }

    /// Large layouts where the old enforcement genuinely failed to converge.
    /// The old swap loop ran all `n` iterations and still left the graph
    /// disconnected; the new repair converges in one O(n + E) pass.
    #[test]
    fn nav_graph_13_large_layouts_converge() {
        let mut layouts: Vec<(&str, Vec<BoundingBox>)> = Vec::new();
        layouts.push(("scatter_1500", {
            let mut rng = Lcg::new(2024);
            (0..1500)
                .map(|_| bb((rng.next_f() * 3600.0) as i32, (rng.next_f() * 2000.0) as i32, 24, 24))
                .collect()
        }));
        layouts.push(("dupes_200", (0..200).map(|_| bb(100, 100, 24, 24)).collect()));
        for (name, boxes) in layouts {
            let fast = NavGraph::build(&boxes);
            assert!(
                is_strongly_connected(&fast.edges, fast.n),
                "{name} (n={}): new graph is not strongly connected",
                fast.n
            );
            let slow = NavGraph::build_oracle(&boxes);
            assert!(
                !is_strongly_connected(&slow.edges, slow.n),
                "{name} (n={}): expected the old enforcement to fail here",
                slow.n
            );
        }
    }
}
