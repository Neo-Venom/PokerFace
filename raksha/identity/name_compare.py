"""M2 Step 2: Separate multi-channel name comparison (never collapsing to one merged key)."""
from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz import fuzz

from raksha.identity.evidence import NameForm, RelativeClue

# Ambiguous Mohammad/Mohd/Md alias family: Md./Mohd. MUST be corroborated by another matching token
_MD_ALIASES = {"mohammad", "mohammed", "muhammad", "muhammed", "mohd", "md"}
_DEVA_RE_CHECK = lambda s: any("\u0900" <= c <= "\u097F" for c in s)


@dataclass(frozen=True)
class NameComparison:
    exact_norm: bool = False
    cross_script: bool = False
    alias_corroborated: bool = False
    initials_corroborated: bool = False
    token_order_match: bool = False
    fuzzy_ratio: float = 0.0
    phonetic_match: bool = False
    given_only_match: bool = False
    strong_disagreement: bool = False

    @property
    def best_level(self) -> str:
        if self.exact_norm or self.cross_script or self.alias_corroborated or self.token_order_match:
            return "exact"
        if self.initials_corroborated or self.fuzzy_ratio >= 86.0:
            return "fuzzy"
        if self.phonetic_match and self.fuzzy_ratio >= 72.0:
            return "phonetic"
        if self.given_only_match:
            return "given_only"
        if self.strong_disagreement:
            return "disagree"
        return "weak"


def _vowel_fold(p: str) -> str:
    """Fold Hindi/Urdu diphthong romanization shifts (ai/e, au/o, ee/i, aa/a) in phonetic keys."""
    s = p.replace("ai", "e").replace("ae", "e").replace("ei", "e").replace("au", "o").replace("ou", "o")
    return s


def _token_match_or_alias(t1: str, t2: str, p1: str, p2: str) -> bool:
    if t1 == t2 or p1 == p2 or _vowel_fold(p1) == _vowel_fold(p2):
        return True
    if t1 in _MD_ALIASES and t2 in _MD_ALIASES:
        return True
    if fuzz.ratio(t1, t2) >= 82.0:
        return True
    return len(p1) >= 4 and len(p2) >= 4 and p1[0] == p2[0] and fuzz.ratio(p1, p2) >= 85.0


def compare_two_names(a: NameForm, b: NameForm) -> NameComparison:
    ta, tb = a.norm_tokens, b.norm_tokens
    pa, pb = a.phonetic_tokens, b.phonetic_tokens
    if not ta or not tb:
        return NameComparison()

    # Single-token fragment vs multi-token name ("Asha" vs "Asha Rajput")
    if len(ta) == 1 or len(tb) == 1:
        single_t, single_p = (ta[0], pa[0]) if len(ta) == 1 else (tb[0], pb[0])
        other_t, other_p = (tb, pb) if len(ta) == 1 else (ta, pa)
        # Never match standalone ambiguous "md" / "mohd"
        if single_t in ("md", "mohd"):
            return NameComparison()
        if (
            single_t == other_t[0]
            or single_p == other_p[0]
            or _vowel_fold(single_p) == _vowel_fold(other_p[0])
            or fuzz.ratio(single_t, other_t[0]) >= 86.0
            or (single_p[0] == other_p[0][0] and fuzz.ratio(single_p, other_p[0]) >= 80.0)
        ):
            return NameComparison(given_only_match=True, fuzzy_ratio=fuzz.ratio(single_t, other_t[0]))
        return NameComparison(strong_disagreement=True)

    # Multi-token full comparison
    sa = " ".join(ta)
    sb = " ".join(tb)
    fuzzy = float(fuzz.token_sort_ratio(sa, sb))

    # 1. Exact normalized token sequence
    if ta == tb:
        cross = _DEVA_RE_CHECK(a.raw) != _DEVA_RE_CHECK(b.raw)
        return NameComparison(exact_norm=not cross, cross_script=cross, fuzzy_ratio=100.0, phonetic_match=True)

    # 2. Cross-script phonetic exact sequence
    if pa == pb and (_DEVA_RE_CHECK(a.raw) != _DEVA_RE_CHECK(b.raw)):
        return NameComparison(cross_script=True, fuzzy_ratio=fuzzy, phonetic_match=True)

    # 3. Token-order permutation (e.g. "Shahid Mohammad" vs "Mohammad Shahid")
    if sorted(ta) == sorted(tb) or sorted(pa) == sorted(pb):
        return NameComparison(token_order_match=True, fuzzy_ratio=fuzzy, phonetic_match=True)

    # 4. Alias match ("Md. Shahid" vs "Mohammad Shahid") - requires corroboration by >=1 other token
    if len(ta) >= 2 and len(tb) >= 2:
        if (ta[0] in _MD_ALIASES and tb[0] in _MD_ALIASES) and _token_match_or_alias(ta[-1], tb[-1], pa[-1], pb[-1]):
            return NameComparison(alias_corroborated=True, fuzzy_ratio=max(fuzzy, 92.0), phonetic_match=True)

    # 5. Initials match ("M. Shahid" vs "Mohammad Shahid" or "K. R. Menon" vs "Kavita R Menon") - corroborated by surname
    if (len(ta[0]) == 1 and tb[0].startswith(ta[0]) or len(tb[0]) == 1 and ta[0].startswith(tb[0])) and _token_match_or_alias(ta[-1], tb[-1], pa[-1], pb[-1]):
        return NameComparison(initials_corroborated=True, fuzzy_ratio=fuzzy, phonetic_match=(pa[-1] == pb[-1]))

    # 6. Phonetic match (candidate/corroborated)
    phon_eq = (pa == pb) or (len(pa) == len(pb) and all(_token_match_or_alias(x, y, px, py) for x, y, px, py in zip(ta, tb, pa, pb)))
    if phon_eq or fuzzy >= 85.0:
        return NameComparison(fuzzy_ratio=fuzzy, phonetic_match=phon_eq)

    # 7. Check strong disagreement (neither first nor last token matches fuzzily/phonetically)
    first_sim = fuzz.ratio(ta[0], tb[0])
    last_sim = fuzz.ratio(ta[-1], tb[-1])
    if first_sim < 68.0 and last_sim < 68.0 and pa[0] != pb[0] and pa[-1] != pb[-1]:
        return NameComparison(fuzzy_ratio=fuzzy, strong_disagreement=True)
    if first_sim < 65.0 and pa[0] != pb[0]:
        return NameComparison(fuzzy_ratio=fuzzy, strong_disagreement=True)

    return NameComparison(fuzzy_ratio=fuzzy, phonetic_match=False)


def compare_name_lists(names_a: list[NameForm], names_b: list[NameForm]) -> NameComparison:
    """Best comparison across all name variants extracted for two records."""
    if not names_a or not names_b:
        return NameComparison()
    order = {"exact": 5, "fuzzy": 4, "phonetic": 3, "given_only": 2, "weak": 1, "disagree": 0}
    best = NameComparison()
    best_rank = -1
    for na in names_a:
        for nb in names_b:
            cmp = compare_two_names(na, nb)
            rk = (order[cmp.best_level], cmp.fuzzy_ratio)
            if best_rank == -1 or rk > (order[best.best_level], best.fuzzy_ratio):
                best = cmp
                best_rank = order[cmp.best_level]
    return best


def compare_relatives(rels_a: list[RelativeClue], rels_b: list[RelativeClue]) -> str:
    """Return 'exact', 'first', 'disagree', or 'missing'."""
    if not rels_a or not rels_b:
        return "missing"
    best = "disagree"
    for ra in rels_a:
        for rb in rels_b:
            # Incompatible relation types when both specified (e.g. D/o vs W/o? Wait: S/o vs W/o implies different gender anyway)
            if ra.norm_tokens == rb.norm_tokens or ra.phonetic_tokens == rb.phonetic_tokens:
                return "exact"
            if len(ra.norm_tokens) >= 1 and len(rb.norm_tokens) >= 1:
                f_sim = fuzz.ratio(ra.norm_tokens[0], rb.norm_tokens[0])
                if ra.phonetic_tokens[0] == rb.phonetic_tokens[0] or f_sim >= 86.0:
                    if len(ra.norm_tokens) >= 2 and len(rb.norm_tokens) >= 2:
                        if ra.phonetic_tokens[-1] == rb.phonetic_tokens[-1] or fuzz.ratio(ra.norm_tokens[-1], rb.norm_tokens[-1]) >= 85.0:
                            return "exact"
                    else:
                        best = "first"
    return best
