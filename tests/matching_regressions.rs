//! Audit regressions for matching correctness (F02–F05, F25).

use audiofp::classical::{
    HaitsmaFingerprint, PanakoFingerprint, PanakoHash, WangFingerprint, WangHash,
};
use audiofp::matching::{
    HaitsmaIndex, HaitsmaMatchConfig, Matcher, PanakoIndex, PanakoMatchConfig, PanakoMatcher,
    WangIndex, WangMatchConfig, WangMatcher, match_ranked, match_result_compare_desc,
};

fn wang_fp(anchors: &[(u32, u32)]) -> WangFingerprint {
    WangFingerprint {
        hashes: anchors
            .iter()
            .map(|&(hash, t)| WangHash { hash, t_anchor: t })
            .collect(),
        frames_per_sec: 62.5,
    }
}

#[test]
fn f02_wang_repeated_excerpt_matches_first_connected_plateau() {
    let q = wang_fp(&(0..8).map(|i| (i, i * 10)).collect::<Vec<_>>());
    let mut r = q.clone();
    r.hashes.extend(q.hashes.iter().map(|h| WangHash {
        hash: h.hash,
        t_anchor: h.t_anchor + 1000,
    }));

    let matcher = WangMatcher::default();
    let direct = matcher.match_one(&q, &r);
    assert!(direct.is_match, "direct matcher must find repeated excerpt");
    assert_eq!(direct.votes, 8);
    assert_eq!(
        direct.offset.frames, 0,
        "first connected plateau wins on ties"
    );

    let indexed = WangIndex::build(&[r], 100)
        .query(&q, &WangMatchConfig::default())
        .map(|(_, m)| m);
    assert_eq!(indexed, Some(direct));
}

#[test]
fn f03_wang_index_jitter_between_occupied_bins() {
    let q = wang_fp(&(0..6).map(|i| (i, 100 + i * 100)).collect::<Vec<_>>());
    let r = WangFingerprint {
        hashes: q
            .hashes
            .iter()
            .enumerate()
            .map(|(i, h)| WangHash {
                hash: h.hash,
                t_anchor: h.t_anchor + if i < 3 { 0 } else { 2 },
            })
            .collect(),
        frames_per_sec: 62.5,
    };

    let matcher = WangMatcher::default();
    let direct = matcher.match_one(&q, &r);
    assert!(direct.is_match, "direct matcher must consolidate jitter");
    assert_eq!(direct.votes, 6);
    assert_eq!(direct.offset.frames, 1);

    let indexed = WangIndex::build(&[r], 100)
        .query(&q, &WangMatchConfig::default())
        .map(|(_, m)| m);
    assert_eq!(indexed, Some(direct));
}

#[test]
fn f04_panako_adjacent_scale_neighbors_not_skipped() {
    let mut q = PanakoFingerprint {
        hashes: Vec::new(),
        frames_per_sec: 62.5,
    };
    let mut r = q.clone();
    let shapes = [
        (1u32, 125u32, 3usize),
        (101, 125, 1),
        (101, 225, 1),
        (201, 225, 3),
    ];
    for (ra, span, n) in shapes {
        for _ in 0..n {
            let hash = q.hashes.len() as u32;
            q.hashes.push(PanakoHash {
                hash,
                t_anchor: 100,
                t_b: 150,
                t_c: 200,
            });
            r.hashes.push(PanakoHash {
                hash,
                t_anchor: ra,
                t_b: ra + span / 2,
                t_c: ra + span,
            });
        }
    }
    for (i, h) in r.hashes.iter_mut().enumerate() {
        let span = h.t_c - h.t_anchor;
        let ra = if i < 3 {
            125
        } else if i == 3 {
            225
        } else if i == 4 {
            125
        } else {
            225
        };
        h.t_anchor = ra;
        h.t_b = ra + span / 2;
        h.t_c = ra + span;
    }

    let cfg = PanakoMatchConfig {
        scale_min: 1.0,
        scale_max: 3.0,
        scale_bins: 2,
        min_votes: 5,
        min_score: 0.01,
        min_prominence: 0.0,
        ransac_refine: false,
        ..Default::default()
    };
    let matcher = PanakoMatcher::new(cfg.clone());
    let with_blockers = matcher.match_one(&q, &r);
    assert!(
        with_blockers.is_match,
        "distant blockers must not hide neighboring evidence"
    );
    assert_eq!(with_blockers.votes, 6);

    q.hashes.retain(|h| h.hash != 3 && h.hash != 4);
    r.hashes.retain(|h| h.hash != 3 && h.hash != 4);

    let direct = matcher.match_one(&q, &r);
    assert!(direct.is_match);
    assert_eq!(direct.votes, 6);
    let indexed = PanakoIndex::build(&[r.clone()], 100)
        .query(&q, &cfg)
        .map(|(_, m)| m);
    assert_eq!(indexed, Some(direct));
}

#[test]
fn f05_stop_hashes_no_resurrection_all_algorithms() {
    let w = WangFingerprint {
        hashes: (0..8)
            .map(|i| WangHash {
                hash: i,
                t_anchor: i * 10,
            })
            .collect(),
        frames_per_sec: 62.5,
    };
    let p = PanakoFingerprint {
        hashes: (0..8)
            .map(|i| PanakoHash {
                hash: i,
                t_anchor: i * 10,
                t_b: i * 10 + 2,
                t_c: i * 10 + 5,
            })
            .collect(),
        frames_per_sec: 62.5,
    };
    let h = HaitsmaFingerprint {
        frames: (0..300).map(|i| i * 7919).collect(),
        frames_per_sec: 78.125,
    };

    let rebuilt_w = WangIndex::build(&vec![w.clone(); 4], 2).query(&w, &WangMatchConfig::default());
    let rebuilt_p =
        PanakoIndex::build(&vec![p.clone(); 4], 2).query(&p, &PanakoMatchConfig::default());
    let rebuilt_h =
        HaitsmaIndex::build(&vec![h.clone(); 4], 2).query(&h, &HaitsmaMatchConfig::default());
    assert!(rebuilt_w.is_none());
    assert!(rebuilt_p.is_none());
    assert!(rebuilt_h.is_none());

    let mut wi = WangIndex::build(&[], 2);
    let mut pi = PanakoIndex::build(&[], 2);
    let mut hi = HaitsmaIndex::build(&[], 2);
    for _ in 0..4 {
        wi.insert(&w, 2);
        pi.insert(&p, 2);
        hi.insert(&h, 2);
    }
    assert!(wi.query(&w, &WangMatchConfig::default()).is_none());
    assert!(pi.query(&p, &PanakoMatchConfig::default()).is_none());
    assert!(hi.query(&h, &HaitsmaMatchConfig::default()).is_none());
}

#[test]
fn f05_stop_hash_remove_and_reinsert_stays_suppressed() {
    let w = WangFingerprint {
        hashes: (0..8)
            .map(|i| WangHash {
                hash: i,
                t_anchor: i * 10,
            })
            .collect(),
        frames_per_sec: 62.5,
    };
    let mut index = WangIndex::build(&[], 2);
    for _ in 0..3 {
        index.insert(&w, 2);
    }
    assert!(index.query(&w, &WangMatchConfig::default()).is_none());
    assert!(index.remove(0));
    assert!(index.remove(1));
    let id = index.insert(&w, 2);
    assert_eq!(id, 1);
    assert!(index.query(&w, &WangMatchConfig::default()).is_none());
}

#[test]
fn f25_nan_scores_rank_without_panic() {
    use audiofp::matching::{MatchResult, TimeOffset};

    let finite = MatchResult {
        is_match: true,
        score: 0.5,
        prominence: 1.0,
        votes: 1,
        offset: TimeOffset::ZERO,
        time_scale: 1.0,
    };
    let nan_score = MatchResult {
        score: f32::NAN,
        prominence: 1.0,
        ..finite
    };
    let nan_prom = MatchResult {
        score: 0.9,
        prominence: f32::NAN,
        ..finite
    };

    let mut ranked = [
        (0, nan_score),
        (1, finite),
        (2, nan_prom),
        (
            3,
            MatchResult {
                score: 0.8,
                prominence: 2.0,
                ..finite
            },
        ),
    ];
    ranked.sort_by(|a, b| match_result_compare_desc(&a.1, &b.1));
    assert_eq!(ranked[0].0, 2, "highest finite score first");
    assert_eq!(ranked[1].0, 3);
    assert_eq!(ranked[2].0, 1);
    assert!(ranked[3].1.score.is_nan());
}

struct NanMatcher;

impl Matcher for NanMatcher {
    type Fingerprint = WangFingerprint;
    type Config = WangMatchConfig;

    fn new(cfg: Self::Config) -> Self {
        let _ = cfg;
        Self
    }

    fn config(&self) -> &Self::Config {
        panic!("not used in this test")
    }

    fn match_one(
        &self,
        _query: &Self::Fingerprint,
        reference: &Self::Fingerprint,
    ) -> audiofp::matching::MatchResult {
        audiofp::matching::MatchResult {
            is_match: true,
            score: f32::NAN,
            prominence: reference.hashes.len() as f32,
            votes: 1,
            offset: audiofp::matching::TimeOffset::ZERO,
            time_scale: 1.0,
        }
    }
}

#[test]
fn f25_match_ranked_survives_custom_nan_matcher() {
    let cfg = WangMatchConfig::default();
    let matcher = NanMatcher::new(cfg);
    let q = wang_fp(&[(0, 10), (1, 20), (2, 30)]);
    let refs = vec![
        wang_fp(&[(0, 10)]),
        wang_fp(&[(0, 10), (1, 20)]),
        wang_fp(&[(0, 10), (1, 20), (2, 30), (3, 40)]),
    ];
    let ranked = match_ranked(&matcher, &q, &refs);
    assert_eq!(ranked.len(), 3);
}
