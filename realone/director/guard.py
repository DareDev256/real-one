"""The literal-lyrics guard: deterministic, no model, no network.

AI music videos get called "too literal" (HN 48940066: "far too much a literal direct copy of the lyrics"; the
classic failure is a line like "watch me move" becoming a wristwatch). This guard rejects a world_prompt that
illustrates its own section's lyric. Two tests, either one rejects:

1. Overlap score: the share of the prompt's content words (stopwords out, simple stemming) that also appear in the
   section's lyric. Over THRESHOLD (0.20) is a re-telling of the line, not a place.
2. Concrete lyric nouns: any depictable noun from the lyric (CONCRETE) that shows up in the prompt, as the same stem,
   as part of a compound ("watch" inside "wristwatch"), or as a literal stand-in (ALIASES: "watch" -> "clock").
   One hit rejects, whatever the score: drawing the object the line names is the literal move.

What it cannot see: a prompt that illustrates the line through words the lyric never uses and the alias table does
not know. Muse Wonder is told the rule; this guard is the part that cannot be talked out of it.
"""
import re

THRESHOLD = 0.20

STOPWORDS = frozenset("""
a about above after again against all almost alone along already also although always am among an and another any
anybody anyhow anymore anyone anything anyway anywhere are around as at away back be became because become been
before behind being below beside besides between beyond both but by came can cannot cant could couldnt did didnt do
does doesnt doing done dont down during each either else enough even ever every everybody everyone everything
everywhere few for from further get gets getting give gives go goes going gone gonna got gotta had hadnt has hasnt
have havent having he hed hell her here hers herself hes him himself his how however i id ill im into is isnt it its
itself ive just keep kind last least less let lets like little lot made make makes many may maybe me might mine more
most much must my myself near need never new next no nobody none nor not nothing now of off often oh ok okay on once
one only onto or other others our ours ourselves out over own past per perhaps put quite rather really right said
same say says see seem seemed seems several shall she shed shell shes should shouldnt since so some somebody someone
something sometimes somewhere still such sure take than that thats the their theirs them themselves then there theres
these they theyd theyll theyre theyve thing things this those though through thru till to too toward towards under
until up upon us very via want wanna was wasnt way we wed well went were werent weve what whats when where whether
which while who whole whom whos whose why will with within without wont would wouldnt ya yall yeah yes yet you youd
youll your youre yours yourself yourselves youve ain aint em cause cuz til tryna
ooh ooo uh huh hey ay ayy whoa woah na nah la da mm mmm hmm yo
""".split())

# Depictable nouns that lyrics name and literal videos draw. Stems (see stem()); kept as data, extend freely.
CONCRETE = frozenset("""
watch clock phone key car truck bus train plane jet boat ship bike motorcycle taxi cab
money cash dollar coin wallet purse bag chain necklace ring bracelet earring diamond gold silver jewel crown throne
gun pistol knife sword bullet blade
rose flower petal tree leaf forest garden field grass
rain snow storm thunder lightning cloud sun moon star rainbow ocean sea wave beach shore river lake waterfall
mountain hill desert island volcano ice fire flame smoke candle lamp lantern match lighter cigarette cigar
bottle glass cup champagne wine whiskey beer liquor drink pill needle
door window wall roof rooftop floor stair staircase elevator bridge tower building skyscraper house mansion castle
church chapel school hospital prison jail cell court bar club casino motel hotel kitchen bedroom bathroom bed couch
sofa chair table desk mirror counter fridge
street road highway alley corner parking subway station airport
mic microphone guitar piano drum speaker radio television tv camera photo picture letter book paper pen envelope
dress shoe sneaker boot jacket coat hoodie hat cap jean shirt mask glove scarf veil
heart hand eye face lip tear blood bone skull tattoo
grave coffin cross angel devil ghost
dog cat bird horse snake wolf lion tiger butterfly spider bee
crown flag balloon cake gift ticket passport suitcase map compass anchor
""".split())

# Literal stand-ins: drawing any of these for a lyric that names the key is still drawing the lyric.
ALIASES = {
    "watch": {"clock", "wristwatch", "rolex", "timepiece", "stopwatch"},
    "clock": {"watch", "wristwatch", "timepiece"},
    "money": {"cash", "dollar", "banknote", "bill", "coin", "stack"},
    "cash": {"money", "dollar", "banknote", "bill"},
    "dollar": {"money", "cash", "banknote", "bill"},
    "car": {"whip", "coupe", "sedan", "convertible", "lambo", "benz", "ferrari", "vehicle"},
    "ride": {"car", "whip", "coupe", "sedan", "convertible"},
    "whip": {"car", "coupe", "sedan", "convertible"},
    "phone": {"cellphone", "smartphone", "iphone", "telephone", "payphone"},
    "gun": {"pistol", "glock", "rifle", "revolver", "firearm"},
    "diamond": {"jewel", "jewelry", "gem", "ice"},
    "ice": {"diamond", "jewel", "jewelry", "gem"},
    "champagne": {"bubbly", "prosecco"},
    "rose": {"flower", "petal", "bouquet"},
    "crown": {"tiara", "throne"},
    "king": {"crown", "throne"},
    "queen": {"crown", "throne", "tiara"},
    "fire": {"flame", "blaze", "inferno", "burn", "bonfire"},
    "rain": {"downpour", "drizzle", "raindrop", "storm"},
    "tear": {"cry", "teardrop", "weep"},
    "moon": {"moonlight", "lunar", "moonlit"},
    "ocean": {"sea", "wave", "beach", "shore"},
    "key": {"keychain", "keyring", "lock"},
    "chain": {"necklace", "pendant"},
    "door": {"doorway", "doorframe"},
    "plane": {"jet", "airplane", "aircraft", "airport"},
    "train": {"subway", "railway", "platform"},
    "heart": {"heartbeat", "valentine"},
}
_ALIAS_CONCRETE = {"ride", "whip", "king", "queen"}  # slang/role words that only matter through their aliases


def tokens(text):
    s = (text or "").lower().replace("’", "'").replace("'", "")
    return [w for w in re.split(r"[^a-z0-9]+", s) if w]


def stem(w):
    """Small, symmetric suffix stripper. Not Porter: it only has to make 'lights'/'light', 'keys'/'key',
    'crying'/'cry', 'burned'/'burn' meet, and it is applied to both sides."""
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 5 and w.endswith("ing"):
        w = w[:-3]
        return w[:-1] if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "lsz" else w
    if len(w) > 4 and w.endswith("ied"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("ed"):
        w = w[:-2]
        return w[:-1] if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "lsz" else w
    if len(w) > 4 and w.endswith("es") and (w[-3] in "sxz" or w[-4:-2] in ("ch", "sh")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss") and not w.endswith("us"):
        return w[:-1]
    if len(w) > 5 and w.endswith("ly"):
        return w[:-2]
    return w


def content_stems(text):
    return [stem(w) for w in tokens(text) if w not in STOPWORDS and len(w) > 2 and not w.isdigit()]


def _concrete_hits(lyric_stems, prompt_stems):
    hits = []
    pset = set(prompt_stems)
    for c in sorted(set(lyric_stems)):
        if c not in CONCRETE and c not in _ALIAS_CONCRETE:
            continue
        how = None
        if c in pset and c in CONCRETE:
            how = c
        else:
            for p in sorted(pset):
                if p == c:
                    continue
                # compound: wrist+watch, night+club, phone+booth. The other half must be a real word-sized piece,
                # so t+rain (train), ter+rain (terrain) and heart+h (hearth) do not count.
                if (len(c) >= 4 and p.endswith(c) and len(p) - len(c) >= 4) or \
                        (len(c) >= 5 and p.startswith(c) and len(p) - len(c) >= 3):
                    how = p
                    break
            if how is None:
                al = sorted(ALIASES.get(c, set()) & pset)
                how = al[0] if al else None
        if how:
            hits.append(c if how == c else f"{c}->{how}")
    return hits


def check(world_prompt, lyric_text, threshold=THRESHOLD):
    """-> {"verdict": "pass"|"reject"|"no_lyric", "score", "threshold", "shared", "concrete", ...}"""
    L = content_stems(lyric_text)
    P = content_stems(world_prompt)
    if not L:
        return {"verdict": "no_lyric", "score": 0.0, "threshold": threshold, "shared": [], "concrete": [],
                "prompt_words": len(set(P)), "lyric_words": 0}
    pset, lset = set(P), set(L)
    shared = sorted(pset & lset)
    score = len(shared) / len(pset) if pset else 0.0
    concrete = _concrete_hits(L, P)
    reject = score > threshold or bool(concrete)
    return {"verdict": "reject" if reject else "pass", "score": round(score, 3), "threshold": threshold,
            "shared": shared, "concrete": concrete, "prompt_words": len(pset), "lyric_words": len(lset)}


def offending_words(report):
    """The words to name back to the model, concrete hits first."""
    words = [c.split("->")[-1] for c in report["concrete"]] + [c.split("->")[0] for c in report["concrete"]]
    words += report["shared"]
    out = []
    for w in words:
        if w not in out:
            out.append(w)
    return out


def check_treatment(treatment, lyrics_by_index, threshold=THRESHOLD):
    """-> {index: report} for every section of a treatment."""
    return {s["index"]: check(s["world_prompt"], "\n".join(lyrics_by_index.get(s["index"], [])), threshold)
            for s in treatment["sections"]}


def reask_message(flagged, treatment, lyrics_by_index):
    """The one re-ask: names each flagged section, its lyric, its score and the exact words to avoid."""
    by = {s["index"]: s for s in treatment["sections"]}
    lines = ["The literal-lyrics guard rejected these world_prompts. Each one draws what the lyric says instead "
             "of where the song is. Rewrite ONLY these sections' world_prompt (and its one-line why) as a place "
             "that carries the subtext you already wrote. Do not use any of the named words, their plurals, or "
             "objects that stand in for them. No people. Return JSON with a `rewrites` array."]
    for i, rep in sorted(flagged.items()):
        lyric = " / ".join(lyrics_by_index.get(i, []))
        lines.append(f"- section {i}: score {rep['score']:.2f} (limit {rep['threshold']:.2f}); "
                     f"avoid: {', '.join(offending_words(rep))}. Lyric: \"{lyric}\". Subtext you wrote: "
                     f"\"{by[i]['subtext']}\". Rejected prompt: \"{by[i]['world_prompt']}\"")
    return "\n".join(lines)
