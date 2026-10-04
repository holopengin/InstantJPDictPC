//! Exact geometric candidate index for the navigation graph (issue #107).
//!
//! `NavGraph::build` used to materialise four sorted candidate lists per node
//! for each of its three phases — O(n²) time and memory, and the memory is the
//! wall (a single n = 6000 scatter build peaks near 2 GB). The consumers,
//! however, only ever look at a handful of the best candidates:
//!
//! * `greedy_assignment` (Phase 1) walks a list from the head for the first
//!   target not already used by another direction.
//! * the Phase 2 and Phase 3 fills do the same for the first target not
//!   already occupied.
//! * the fill *ordering* compares the number of available candidates per
//!   unfilled direction (a full-list count, not a prefix).
//!
//! Both walkers never need more than the top **five** candidates, and that is a
//! proof, not an observation: at the moment a walker scans a list it holds at
//! most four distinct used/occupied targets, so among any five distinct
//! candidates at least one is free; when the list has fewer than five entries
//! a five-candidate read is the whole list, so an empty answer means the list
//! is genuinely exhausted. (`k = 5` therefore reproduces the old `for j in
//! 0..n` fallback exactly, without ever scanning.) The fill ordering does need
//! the exact full count, which this module answers with a range query.
//!
//! This module replaces the materialised lists with:
//!
//! * a k-d tree over the normalised centres, and a best-first traversal that
//!   yields the exact top-`k` candidates in the phase's own order (with the
//!   exact `f32` comparator and tie-breaks), and
//! * a pruned range count for the fill ordering.
//!
//! Exactness: the traversal's heap keys for *points* are the exact cost tuples
//! the old comparator produced, so the emitted order is identical. Internal
//! nodes carry a lower bound (computed in `f64`, minus a slack that dwarfs the
//! `f32` rounding of the true cost) so a contained point can never be emitted
//! before a strictly cheaper one. The public surface returns indices, never
//! cost values, so consumers cannot observe the relaxation.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Off-axis penalty weight, Phase 1 (strict local).
pub(crate) const W: f32 = 10.0;
/// Off-axis penalty weight, Phases 2 and 3 (island connecting / wrap fill).
pub(crate) const W2: f32 = 1.5;
/// Max Euclidean distance for Phase 1.
pub(crate) const LOCAL_DIST: f32 = 0.05;
/// Allow within 45° of cardinal (`off_axis <= primary`).
pub(crate) const CONE45: f32 = 1.0;

/// Slack subtracted from every internal-node lower bound. The true cost is an
/// `f32` expression whose rounding error is at most a few ULPs (relative
/// ~1e-7); costs live in `[0, 2.5]`, so `1e-4` makes the bound conservative by
/// orders of magnitude. The slack only widens the search, never reorders the
/// emitted points, whose keys are exact.
const LB_SLACK: f64 = 1e-4;

/// Leaf bucket size for the k-d tree.
const LEAF: usize = 8;

#[inline]
fn torus_dx(x1: f32, x2: f32) -> f32 {
    let raw = (x1 - x2).abs();
    raw.min(1.0 - raw)
}

#[inline]
fn torus_dy(y1: f32, y2: f32) -> f32 {
    let raw = (y1 - y2).abs();
    raw.min(1.0 - raw)
}

/// Whether `j` is a candidate for direction `dir` of node `i` in `phase`
/// (`1`, `2` or `3`). Exactly the predicate the old per-phase scan applied,
/// with the Phase-1 radius check folded in via the precomputed `euc`.
#[inline]
pub(crate) fn in_region(
    phase: u8,
    dir: u8,
    xi: f32,
    yi: f32,
    xj: f32,
    yj: f32,
    euc: f32,
) -> bool {
    let xd = (xi - xj).abs();
    let yd = (yi - yj).abs();
    match phase {
        1 => {
            if euc > LOCAL_DIST {
                return false;
            }
            match dir {
                0 => yj < yi && xd <= CONE45 * (yi - yj),
                1 => yj > yi && xd <= CONE45 * (yj - yi),
                2 => xj > xi && yd <= CONE45 * (xj - xi),
                _ => xj < xi && yd <= CONE45 * (xi - xj),
            }
        }
        2 => match dir {
            0 => yj < yi && xd <= CONE45 * (yi - yj),
            1 => yj > yi && xd <= CONE45 * (yj - yi),
            2 => xj > xi && yd <= CONE45 * (xj - xi),
            _ => xj < xi && yd <= CONE45 * (xi - xj),
        },
        _ => match dir {
            0 => {
                if yj <= yi {
                    false
                } else {
                    let dy = (yi - yj + 1.0) % 1.0;
                    xd <= CONE45 * dy
                }
            }
            1 => {
                if yj >= yi {
                    false
                } else {
                    let dy = (yj - yi + 1.0) % 1.0;
                    xd <= CONE45 * dy
                }
            }
            2 => {
                if xj >= xi {
                    false
                } else {
                    let dx = (xj - xi + 1.0) % 1.0;
                    yd <= CONE45 * dx
                }
            }
            _ => {
                if xj <= xi {
                    false
                } else {
                    let dx = (xi - xj + 1.0) % 1.0;
                    yd <= CONE45 * dx
                }
            }
        },
    }
}

/// The exact `(cost, euc)` the old phase computed for candidate `j` of node
/// `i`. `euc` is the Phase-1 euclidean tie-break (`0.0` elsewhere).
#[inline]
pub(crate) fn cost_of(phase: u8, dir: u8, xi: f32, yi: f32, xj: f32, yj: f32) -> (f32, f32) {
    let xd = (xi - xj).abs();
    let yd = (yi - yj).abs();
    let euc = (xd * xd + yd * yd).sqrt();
    let c = match phase {
        1 => match dir {
            0 => (yi - yj) + W * xd,
            1 => (yj - yi) + W * xd,
            2 => (xj - xi) + W * yd,
            _ => (xi - xj) + W * yd,
        },
        2 => match dir {
            0 => (yi - yj) + W2 * xd,
            1 => (yj - yi) + W2 * xd,
            2 => (xj - xi) + W2 * yd,
            _ => (xi - xj) + W2 * yd,
        },
        _ => {
            let tx = torus_dx(xi, xj);
            let ty = torus_dy(yi, yj);
            match dir {
                0 => ((yi - yj + 1.0) % 1.0) + W2 * tx,
                1 => ((yj - yi + 1.0) % 1.0) + W2 * tx,
                2 => ((xj - xi + 1.0) % 1.0) + W2 * ty,
                _ => ((xi - xj + 1.0) % 1.0) + W2 * ty,
            }
        }
    };
    (c, euc)
}

#[derive(Clone)]
struct Node {
    aabb: [f32; 4], // x0, y0, x1, y1
    min_idx: u32,
    lo: u32,
    hi: u32,
    axis: u8,
    left: u32,
    right: u32,
    leaf: bool,
}

struct Builder<'a> {
    pos: &'a [(f32, f32)],
    order: Vec<u32>,
    nodes: Vec<Node>,
}

impl<'a> Builder<'a> {
    fn aabb_of(&self, lo: usize, hi: usize) -> ([f32; 4], u32) {
        let mut a = [f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY];
        let mut mi = u32::MAX;
        for &p in &self.order[lo..hi] {
            let (x, y) = self.pos[p as usize];
            if x < a[0] { a[0] = x; }
            if y < a[1] { a[1] = y; }
            if x > a[2] { a[2] = x; }
            if y > a[3] { a[3] = y; }
            if p < mi { mi = p; }
        }
        (a, mi)
    }

    fn build_range(&mut self, lo: usize, hi: usize) -> u32 {
        let (aabb, min_idx) = self.aabb_of(lo, hi);
        let me = self.nodes.len() as u32;
        self.nodes.push(Node {
            aabb,
            min_idx,
            lo: lo as u32,
            hi: hi as u32,
            axis: 0,
            left: u32::MAX,
            right: u32::MAX,
            leaf: true,
        });
        if hi - lo <= LEAF {
            return me;
        }
        let ex = aabb[2] - aabb[0];
        let ey = aabb[3] - aabb[1];
        let axis = if ex >= ey { 0u8 } else { 1u8 };
        let pos: &[(f32, f32)] = self.pos;
        let order = &mut self.order[lo..hi];
        order.sort_by(|&a, &b| {
            let (ax, ay) = pos[a as usize];
            let (bx, by) = pos[b as usize];
            let (k1, k2) = if axis == 0 { (ax, bx) } else { (ay, by) };
            k1.partial_cmp(&k2)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });
        let mid = lo + (hi - lo) / 2;
        let left = self.build_range(lo, mid);
        let right = self.build_range(mid, hi);
        let n = &mut self.nodes[me as usize];
        n.leaf = false;
        n.axis = axis;
        n.left = left;
        n.right = right;
        me
    }
}

/// A k-d tree over the normalised centres, with exact top-k and count queries
/// for the three build phases.
pub(crate) struct NavIndex {
    pos: Vec<(f32, f32)>,
    order: Vec<u32>,
    nodes: Vec<Node>,
}

/// Heap entry: either an internal node (with a lower-bound key) or a single
/// point (with its exact key).
#[derive(Clone, Copy)]
struct Key {
    cost: f32,
    aux: f32,
    idx: u32,
    node: u32,
    is_point: bool,
}

impl Key {
    #[inline]
    fn cmp_lex(&self, other: &Key, phase1: bool) -> Ordering {
        match self.cost.partial_cmp(&other.cost).unwrap_or(Ordering::Equal) {
            Ordering::Equal => {}
            o => return o,
        }
        if phase1 {
            match self.aux.partial_cmp(&other.aux).unwrap_or(Ordering::Equal) {
                Ordering::Equal => {}
                o => return o,
            }
        }
        match self.idx.cmp(&other.idx) {
            Ordering::Equal => {}
            o => return o,
        }
        // Node before point on a full tie; only affects expansion order.
        (!self.is_point).cmp(&(!other.is_point))
    }
}

/// `Ordering` wrapper so `Key` can live in a `BinaryHeap`. The phase-1 flag is
/// carried in `aux` semantics; `Ord` is only ever consulted through
/// [`Key::cmp_lex`], so implement it as the lex comparison with `phase1 =
/// true` won't do — instead the heap uses a fixed comparison and the caller
/// encodes phase in `aux`. See `top_k`.
struct HeapKey(Key, bool);

impl PartialEq for HeapKey {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for HeapKey {}
impl PartialOrd for HeapKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapKey {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reversed for a min-heap (`BinaryHeap` is a max-heap).
        other.0.cmp_lex(&self.0, self.1)
    }
}

#[inline]
fn clamp_dist(v: f32, lo: f32, hi: f32) -> f32 {
    if v < lo {
        lo - v
    } else if v > hi {
        v - hi
    } else {
        0.0
    }
}

/// Minimum torus distance from `v` to the interval `[lo, hi]`.
#[inline]
fn torus_dist_min(v: f32, lo: f32, hi: f32) -> f32 {
    if v >= lo && v <= hi {
        0.0
    } else {
        torus_dx(v, lo).min(torus_dx(v, hi))
    }
}

impl NavIndex {
    pub(crate) fn new(pos: &[(f32, f32)]) -> Self {
        let mut b = Builder {
            pos,
            order: (0..pos.len() as u32).collect(),
            nodes: Vec::new(),
        };
        if !pos.is_empty() {
            b.build_range(0, pos.len());
        }
        NavIndex {
            pos: pos.to_vec(),
            order: b.order,
            nodes: b.nodes,
        }
    }

    #[inline]
    pub(crate) fn position(&self, i: usize) -> (f32, f32) {
        self.pos[i]
    }

    /// Whether any point of the box could satisfy the region predicate. A
    /// conservative test: it may say `true` for an empty box, but never
    /// `false` for a box that contains a candidate.
    #[inline]
    fn region_possible(phase: u8, dir: u8, xi: f32, yi: f32, a: &[f32; 4]) -> bool {
        let (x0, y0, x1, y1) = (a[0], a[1], a[2], a[3]);
        match phase {
            1 | 2 => match dir {
                0 => {
                    if y0 >= yi {
                        return false;
                    }
                    // max cone half-width over the box is yi - y0.
                    x1 >= xi - (yi - y0) && x0 <= xi + (yi - y0)
                }
                1 => {
                    if y1 <= yi {
                        return false;
                    }
                    x1 >= xi - (y1 - yi) && x0 <= xi + (y1 - yi)
                }
                2 => {
                    if x1 <= xi {
                        return false;
                    }
                    y1 >= yi - (x1 - xi) && y0 <= yi + (x1 - xi)
                }
                _ => {
                    if x0 >= xi {
                        return false;
                    }
                    y1 >= yi - (xi - x0) && y0 <= yi + (xi - x0)
                }
            },
            _ => match dir {
                0 => {
                    if y1 <= yi {
                        return false;
                    }
                    // Widest cone in the box is at the y closest to yi (> yi).
                    let w = 1.0 - (y0 - yi).max(0.0);
                    x1 >= xi - w && x0 <= xi + w
                }
                1 => {
                    if y0 >= yi {
                        return false;
                    }
                    let w = 1.0 - (yi - y1).max(0.0);
                    x1 >= xi - w && x0 <= xi + w
                }
                2 => {
                    if x0 >= xi {
                        return false;
                    }
                    let w = 1.0 - (xi - x1).max(0.0);
                    y1 >= yi - w && y0 <= yi + w
                }
                _ => {
                    if x1 <= xi {
                        return false;
                    }
                    let w = 1.0 - (x0 - xi).max(0.0);
                    y1 >= yi - w && y0 <= yi + w
                }
            },
        }
    }

    /// Lower-bound key `(cost, euc)` for a box, in the phase's own order. The
    /// result is conservative: for every point `p` in the box,
    /// `(lb_cost, lb_euc) <= (cost(p), euc(p))` lexicographically.
    #[inline]
    fn lower_bound(phase: u8, dir: u8, xi: f32, yi: f32, a: &[f32; 4]) -> (f32, f32) {
        let (x0, y0, x1, y1) = (a[0], a[1], a[2], a[3]);
        let (lc, le): (f64, f64);
        match phase {
            1 | 2 => {
                let w = if phase == 1 { W as f64 } else { W2 as f64 };
                match dir {
                    0 => {
                        let yc = y1.min(yi);
                        let primary = (yi - yc).max(0.0) as f64;
                        let off = clamp_dist(xi, x0, x1) as f64;
                        lc = primary + w * off;
                    }
                    1 => {
                        let yc = y0.max(yi);
                        let primary = (yc - yi).max(0.0) as f64;
                        let off = clamp_dist(xi, x0, x1) as f64;
                        lc = primary + w * off;
                    }
                    2 => {
                        let xc = x0.max(xi);
                        let primary = (xc - xi).max(0.0) as f64;
                        let off = clamp_dist(yi, y0, y1) as f64;
                        lc = primary + w * off;
                    }
                    _ => {
                        let xc = x1.min(xi);
                        let primary = (xi - xc).max(0.0) as f64;
                        let off = clamp_dist(yi, y0, y1) as f64;
                        lc = primary + w * off;
                    }
                }
                let dx = clamp_dist(xi, x0, x1) as f64;
                let dy = clamp_dist(yi, y0, y1) as f64;
                le = (dx * dx + dy * dy).sqrt();
            }
            _ => {
                match dir {
                    0 => {
                        let yc = y1.max(yi);
                        let dy = (1.0 - (yc - yi) as f64).max(0.0);
                        let tx = torus_dist_min(xi, x0, x1) as f64;
                        lc = dy + W2 as f64 * tx;
                    }
                    1 => {
                        let yc = y0.min(yi);
                        let dy = (1.0 - (yi - yc) as f64).max(0.0);
                        let tx = torus_dist_min(xi, x0, x1) as f64;
                        lc = dy + W2 as f64 * tx;
                    }
                    2 => {
                        let xc = x0.min(xi);
                        let dx = (1.0 - (xi - xc) as f64).max(0.0);
                        let ty = torus_dist_min(yi, y0, y1) as f64;
                        lc = dx + W2 as f64 * ty;
                    }
                    _ => {
                        let xc = x1.max(xi);
                        let dx = (1.0 - (xc - xi) as f64).max(0.0);
                        let ty = torus_dist_min(yi, y0, y1) as f64;
                        lc = dx + W2 as f64 * ty;
                    }
                }
                le = 0.0;
            }
        }
        ((lc - LB_SLACK) as f32, (le - LB_SLACK).max(0.0) as f32)
    }

    /// Exact top-`k` candidates for `(phase, dir)` at node `i`, in the phase's
    /// order, excluding `excluded` (and `i` itself). `k` is `5` at every call
    /// site; a proof that five suffices is in the module header.
    pub(crate) fn top_k(
        &self,
        i: usize,
        phase: u8,
        dir: u8,
        excluded: &[usize],
        k: usize,
    ) -> Vec<usize> {
        let n = self.pos.len();
        if n == 0 || self.nodes.is_empty() {
            return Vec::new();
        }
        let phase1 = phase == 1;
        let (xi, yi) = self.pos[i];
        let mut heap: BinaryHeap<HeapKey> = BinaryHeap::new();
        let root = &self.nodes[0];
        let (lc, le) = Self::lower_bound(phase, dir, xi, yi, &root.aabb);
        heap.push(HeapKey(
            Key { cost: lc, aux: le, idx: root.min_idx, node: 0, is_point: false },
            phase1,
        ));

        let mut out: Vec<usize> = Vec::with_capacity(k);
        while out.len() < k {
            let Some(HeapKey(e, _)) = heap.pop() else { break };
            if e.is_point {
                out.push(e.node as usize);
                continue;
            }
            let node = &self.nodes[e.node as usize];
            if node.leaf {
                for &p in &self.order[node.lo as usize..node.hi as usize] {
                    let pi = p as usize;
                    if pi == i || excluded.contains(&pi) || out.contains(&pi) {
                        continue;
                    }
                    let (xj, yj) = self.pos[pi];
                    let xd = (xi - xj).abs();
                    let yd = (yi - yj).abs();
                    let euc = (xd * xd + yd * yd).sqrt();
                    if !in_region(phase, dir, xi, yi, xj, yj, euc) {
                        continue;
                    }
                    let (c, ec) = cost_of(phase, dir, xi, yi, xj, yj);
                    heap.push(HeapKey(
                        Key { cost: c, aux: ec, idx: p, node: p, is_point: true },
                        phase1,
                    ));
                }
            } else {
                for &child_id in &[node.left, node.right] {
                    let child = &self.nodes[child_id as usize];
                    if !Self::region_possible(phase, dir, xi, yi, &child.aabb) {
                        continue;
                    }
                    let (lc, le) = Self::lower_bound(phase, dir, xi, yi, &child.aabb);
                    heap.push(HeapKey(
                        Key {
                            cost: lc,
                            aux: le,
                            idx: child.min_idx,
                            node: child_id,
                            is_point: false,
                        },
                        phase1,
                    ));
                }
            }
        }
        out
    }

    /// Exact number of candidates for `(phase, dir)` at node `i` (excluding
    /// `i`), matching the old `list.iter().filter(|(v,_)| v != &i).count()`.
    pub(crate) fn count(&self, i: usize, phase: u8, dir: u8) -> usize {
        if self.nodes.is_empty() {
            return 0;
        }
        let (xi, yi) = self.pos[i];
        let mut total = 0usize;
        let mut stack = vec![0u32];
        while let Some(nid) = stack.pop() {
            let node = &self.nodes[nid as usize];
            if !Self::region_possible(phase, dir, xi, yi, &node.aabb) {
                continue;
            }
            if node.leaf {
                for &p in &self.order[node.lo as usize..node.hi as usize] {
                    let pi = p as usize;
                    if pi == i {
                        continue;
                    }
                    let (xj, yj) = self.pos[pi];
                    let xd = (xi - xj).abs();
                    let yd = (yi - yj).abs();
                    let euc = (xd * xd + yd * yd).sqrt();
                    if in_region(phase, dir, xi, yi, xj, yj, euc) {
                        total += 1;
                    }
                }
            } else {
                stack.push(node.left);
                stack.push(node.right);
            }
        }
        total
    }
}
