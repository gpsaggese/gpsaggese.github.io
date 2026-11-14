"""Conservative, standard-library event extraction; no models or return forecasts.

Evidence offsets address the original ``headline`` / ``text`` strings. Dates are
source timestamps, not proof that a historical trading system saw the article.
``reported_fact`` means that the source reports a fact, not independent verification.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import re
from zoneinfo import ZoneInfo


COMPANY_NAMES = {
    "NVDA": ("Nvidia",), "TSLA": ("Tesla",), "PLTR": ("Palantir",),
    "SNDK": ("SanDisk", "Sandisk"), "AMD": ("Advanced Micro Devices",),
    "AVGO": ("Broadcom",), "MU": ("Micron",), "MSFT": ("Microsoft",),
    "AAPL": ("Apple",), "GOOGL": ("Alphabet", "Google"),
    "GOOG": ("Alphabet", "Google"), "AMZN": ("Amazon", "AWS", "Amazon Web Services"), "META": ("Meta Platforms",),
    "ORCL": ("Oracle",), "CRM": ("Salesforce",), "JPM": ("JPMorgan", "JPMorgan Chase"),
    "BAC": ("Bank of America",), "XOM": ("Exxon", "ExxonMobil"),
    "CVX": ("Chevron",), "UNH": ("UnitedHealth",), "LLY": ("Eli Lilly",),
}
BROKERS = {
    "bank_of_america": r"\b(?:BofA|Bank of America)\b",
    "morgan_stanley": r"\bMorgan Stanley\b", "goldman_sachs": r"\bGoldman(?: Sachs)?\b",
    "jpmorgan": r"\bJ\.?P\.?\s?Morgan(?: Chase)?\b", "citi": r"\bCiti(?:group)?\b",
    "barclays": r"\bBarclays\b", "ubs": r"\bUBS\b", "jefferies": r"\bJefferies\b",
    "deutsche_bank": r"\bDeutsche Bank\b", "wells_fargo": r"\bWells Fargo\b",
    "rbc": r"\bRBC\b", "mizuho": r"\bMizuho\b", "bernstein": r"\bBernstein\b",
    "piper_sandler": r"\bPiper Sandler\b", "wedbush": r"\bWedbush\b",
    "baird": r"\bBaird\b", "bmo": r"\bBMO\b", "needham": r"\bNeedham\b",
    "stifel": r"\bStifel\b", "oppenheimer": r"\bOppenheimer\b",
}

_PATTERNS = (
    ("analyst_revision", r"\b(?:price[ -]target|analyst|upgrad\w*|downgrad\w*|reiterat\w+|initiat\w* coverage|maintain\w* (?:a |an )?(?:buy|sell|hold|outperform|underperform))\b"),
    ("contract_change", r"\b(?:non[ -]?renew\w*|not (?:be )?renew\w*|(?:did|does|will) not renew|terminat\w+|cancel\w+)\b.{0,100}\b(?:contract|agreement|deal)\b|\b(?:contract|agreement|deal)\b.{0,100}\b(?:non[ -]?renew\w*|not (?:be )?renew\w*|terminat\w+|cancel\w+)\b"),
    ("financial_results", r"\b(?:earnings|quarterly results|financial results|quarter revenue|quarter profit|revenue guidance|profit forecast|full[ -]year (?:outlook|guidance))\b|\b(?:reports?|posts?)\b.{0,40}\b(?:revenue|profit|results)\b"),
    ("partnership", r"\b(?:partners? with|partnership|collaborat\w+|strategic alliance|joint venture)\b"),
    ("product", r"\b(?:launch\w*|unveil\w*|introduc\w*|releases? (?:a |an |its )?(?:new |updated )?(?:product|chip|model|software)|product announcement)\b"),
    ("contract_change", r"\b(?:contract|agreement|deal|cloud credits?|compute credits?)\b"),
    ("trading_activity", r"\b(?:shares?|stock)\b.{0,50}\b(?:rose|fell|rise\w*|fall\w*|gain\w*|drop\w*|rall\w*|slid\w*|surge\w*|trade\w*)\b|\b(?:bought|sold|selling|buying|trimmed|added|reduced|increased)\b.{0,60}\b(?:shares?|stake|position|holding)\b|\b(?:premarket|after[ -]hours|short interest|options volume)\b"),
    ("market_commentary", r"\b(?:market|sector|industry|outlook|bullish|bearish|inflation|treasury|oil prices?|rally|selloff)\b"),
)
_UNCERTAIN = re.compile(r"\b(?:reportedly|rumou?r\w*|unconfirmed|could|might|may|in talks|sources? (?:say|said)|according to sources|denies?|not confirmed|(?:did|has|have|does|will|is|was) not (?:sign\w*|announc\w*|launch\w*|win|award\w*|agree\w*))\b", re.I)
_CREDITS = re.compile(r"\b(?:cloud|compute|computing)\b.{0,40}\bcredits?\b|\bcredits?\b.{0,40}\b(?:cloud|compute|computing)\b", re.I)
_NEGATED_STANCE = re.compile(r"\b(?:not|never|no longer)\s+(?:\w+\s+){0,2}(?:bullish|bearish|overweight|underweight|outperform|underperform|buy|sell)\b", re.I)
_RATINGS = re.compile(r"\b(?:buy|sell|hold|outperform|underperform|overweight|underweight|neutral|market perform|equal weight)\b", re.I)
_STOP = set("a an the of to from for in on at by with and or its it is are was were has have had as that this said says reports reported new shares stock stocks company firms analyst analysts price target rating ratings raises raised cuts cut lowers lowered maintains maintained reiterates reiterated upgrades upgraded downgrades downgraded".split())


def _string(value):
    return value if isinstance(value, str) else ""


def _time(value):
    try:
        parsed = datetime.fromisoformat(_string(value).replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else None
    except ValueError:
        return None


def _report_day(stamp):
    return stamp.astimezone(ZoneInfo("America/New_York")).date()


def _mention_pattern(symbol):
    aliases = COMPANY_NAMES.get(symbol, ())
    names = "|".join(re.escape(name) for name in aliases)
    # Ticker matching is case-sensitive: ordinary words such as "mu" are not tags.
    ticker = rf"(?<![A-Za-z0-9])\$?{re.escape(symbol)}(?![A-Za-z0-9])"
    return re.compile(ticker + (rf"|(?i:\b(?:{names})\b)" if names else ""))


def _units(article, mention):
    units = []
    # Clause boundaries reduce transfer of an event from a different company.
    boundary = re.compile(r"(?<=[.!?;])\s+|\n+|\b(?:while|whereas|but)\b", re.I)
    for field in ("headline", "text"):
        original = _string(article.get(field))
        start = 0
        for split in list(boundary.finditer(original)) + [None]:
            if split and split.start() and original[split.start() - 1] == ".":
                prefix = original[:split.start()]
                if re.search(r"\b(?:Inc|Corp|Co|Ltd|Mr|Mrs|Ms|Dr|St|vs)\.$|(?:\b[A-Za-z]\.){2,}$", prefix, re.I) or re.search(r"\b[A-Z]\.$", prefix):
                    continue
            end = split.start() if split else len(original)
            left, right = start, end
            while left < right and original[left].isspace():
                left += 1
            while right > left and original[right - 1].isspace():
                right -= 1
            text = original[left:right]
            if mention.search(text):
                units.append((field, left, right, text))
            start = split.end() if split else end
    return units


def company_clauses(article: dict, symbol: str) -> list[str]:
    """Return literal company-containing clauses for a separate NLP consumer.

    These are lexical scopes, not guaranteed semantic attribution. No pronoun
    resolution or implied transfer from a different company's clause is applied.
    """
    symbol = _string(symbol).strip().upper()
    if not symbol:
        return []
    return [unit[3] for unit in _units(article, _mention_pattern(symbol))]


def _classification(text):
    for kind, pattern in _PATTERNS:
        hit = re.search(pattern, text, re.I)
        if hit:
            return kind, hit
    return "other", None


def _action(text, kind):
    if kind == "analyst_revision":
        for action, pattern in (
            ("upgrade", r"\bupgrad\w*"), ("downgrade", r"\bdowngrad\w*"),
            ("raise_target", r"\b(?:rais\w*|lift\w*|boost\w*)\b.{0,65}\b(?:target|PT)\b"),
            ("lower_target", r"\b(?:cut\w*|lower\w*|reduc\w*)\b.{0,65}\b(?:target|PT)\b"),
            ("initiate", r"\binitiat\w*"), ("reiterate", r"\b(?:reiterat\w*|maintain\w*)"),
        ):
            if re.search(pattern, text, re.I):
                return action
        return "analyst_comment"
    if re.search(r"\b(?:non[ -]?renew\w*|not (?:be )?renew\w*|(?:did|does|will) not renew|terminat\w*|cancel\w*)\b", text, re.I):
        return "termination_or_nonrenewal"
    if _CREDITS.search(text):
        return "cloud_credits"
    if re.search(r"\b(?:not signed|denies?|unconfirmed|in talks)\b", text, re.I):
        return "unconfirmed"
    if re.search(r"\b(?:wins?|won|awarded|signs?|signed|secures?)\b", text, re.I):
        return "agreement_reported"
    return kind


def _metadata(text, symbol, kind):
    aliases = set(re.findall(r"[a-z0-9]+", " ".join(COMPANY_NAMES.get(symbol, ())).lower()))
    tokens = set(re.findall(r"[a-z][a-z0-9]+", text.lower())) - _STOP - aliases - {symbol.lower()}
    brokers = tuple(sorted(k for k, pattern in BROKERS.items() if re.search(pattern, text, re.I)))
    numbers = tuple(sorted(set(re.findall(r"\$\s*\d[\d,.]*(?:\s*(?:billion|million|bn|m))?|\b\d+(?:\.\d+)?%|\bQ[1-4]\b|\b20\d{2}\b", text, re.I))))
    return tokens, brokers, numbers, _action(text, kind)


def _analyst_details(text):
    brokers = tuple(sorted(k for k, pattern in BROKERS.items() if re.search(pattern, text, re.I)))
    action = _action(text, "analyst_revision")
    rating_words = r"buy|sell|hold|neutral|outperform|underperform|overweight|underweight|equal[ -]weight|market perform"
    rating = re.search(rf"\b(?:upgrad\w*|downgrad\w*|initiat\w*)\b.{{0,140}}?\bto\s+(?:a\s+|an\s+)?({rating_words})\b", text, re.I)
    if not rating:
        rating = re.search(rf"\b(?:reiterat\w*|maintain\w*|initiat\w*)\b.{{0,100}}?\b({rating_words})\b", text, re.I)
    destination = rating.group(1).lower().replace("-", " ") if rating else None
    # Prefer the new value in "target from $old to $new".
    target = re.search(r"\b(?:price[ -]target|target price|PT)\s+from\s*\$[\d,.]+\s+to\s*\$([\d,.]+)", text, re.I)
    if not target:
        target = re.search(r"\b(?:price[ -]target|target price|PT)\s*(?:(?:of|at|to|is|was)\s*)?\$([\d,.]+)", text, re.I)
    if not target:
        target = re.search(r"\$([\d,.]+)\s+(?:price[ -]target|target price|PT)\b", text, re.I)
    target_value = None
    if target:
        try:
            target_value = format(float(target.group(1).replace(",", "")), "g")
        except ValueError:
            pass
    return brokers, action, destination, target_value


def _analyst_summary(context):
    brokers, action, rating, target = _analyst_details(context)
    labels = {"bank_of_america": "Bank of America", "jpmorgan": "JPMorgan", "ubs": "UBS", "rbc": "RBC", "bmo": "BMO"}
    actor = ", ".join(labels.get(b, b.replace("_", " ").title()) for b in brokers) or "An analyst"
    verb = {"upgrade": "upgraded the stock", "downgrade": "downgraded the stock", "raise_target": "raised the price target",
            "lower_target": "lowered the price target", "initiate": "initiated coverage", "reiterate": "reiterated a rating or target"}.get(action, "commented on the stock")
    detail = f" to {rating.title()}" if rating and action in ("upgrade", "downgrade") else f"; the cited rating is {rating.title()}" if rating else ""
    if target:
        detail += f"; the cited price target is ${target}"
    return f"{actor} {verb}{detail}. Analyst opinions are not return forecasts."


def _evidence(article, unit, hit, mention):
    field, start, end, text = unit
    if len(text) <= 360:
        return [{"field": field, "start": start, "end": end, "text": text}]
    result = []
    for match in (mention.search(text), hit):
        if match:
            left = max(0, match.start() - 70)
            right = min(len(text), max(match.end() + 70, left + 140))
            item = {"field": field, "start": start + left, "end": start + right,
                    "text": _string(article.get(field))[start + left:start + right]}
            if item not in result:
                result.append(item)
    return result


def extract_article(article: dict, symbol: str) -> dict:
    """Extract a single-company record with verbatim evidence and no prediction.

    ``stance`` is bullish/bearish/mixed/unknown and only reflects explicit
    analyst or opinion wording. Product launches and contract values do not
    automatically receive an investment stance.
    """
    symbol = _string(symbol).strip().upper()
    if not symbol or not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,14}", symbol):
        raise ValueError("symbol must be a nonempty stock ticker")
    mention = _mention_pattern(symbol)
    units = _units(article, mention)
    tags = article.get("symbols") or []
    if isinstance(tags, str):
        tags = [tags]
    tagged = symbol in {_string(tag).upper() for tag in tags}
    relevance = "headline" if any(u[0] == "headline" for u in units) else "body" if units else "tag_only" if tagged else "none"
    limitations = ["Rules match wording; source claims are not independently verified.",
                   "No return prediction or calibrated probability is produced."]
    selected, kind, hit = None, "other", None
    for field in ("headline", "text"):
        candidates = [(unit, *_classification(unit[3])) for unit in units if unit[0] == field]
        recognized = [candidate for candidate in candidates if candidate[1] != "other"]
        if recognized:
            selected, kind, hit = recognized[0]
            break
    if selected is None and units:
        selected = units[0]
    if kind == "analyst_revision":
        # A roundup headline can name the company without naming its broker.
        # Prefer the explicit company-specific action in the body in that case.
        candidates = [(unit, *_classification(unit[3])) for unit in units]
        candidates = [candidate for candidate in candidates if candidate[1] == "analyst_revision"]
        def detail_score(candidate):
            brokers, action, rating, target = _analyst_details(candidate[0][3])
            return (bool(brokers), action != "analyst_comment", bool(rating), bool(target), candidate[0][0] == "headline")
        selected, kind, hit = max(candidates, key=detail_score)
    context = selected[3] if selected else ""
    evidence = _evidence(article, selected, hit, mention) if selected else []
    action = _action(context, kind)
    uncertain = bool(_UNCERTAIN.search(context))
    if kind == "analyst_revision" and uncertain:
        # "Upgraded ... demand could grow" reports a rating while making the
        # business outlook conditional. Do not turn that rating into a rumor.
        rating_unconfirmed = re.search(
            r"\b(?:reportedly|rumou?r\w*|unconfirmed|denies?|not confirmed|sources? (?:say|said)|according to sources)\b"
            r"|\b(?:could|might|may|would)\s+(?:\w+\s+){0,2}(?:upgrad\w*|downgrad\w*|rais\w*|lower\w*|cut\w*|initiat\w*|reiterat\w*)\b",
            context, re.I,
        )
        if not rating_unconfirmed:
            uncertain = False
            limitations.append("Conditional outlook wording does not establish future business outcomes; the reported rating remains an analyst opinion.")
    stance = "unknown"
    if kind in ("analyst_revision", "market_commentary"):
        positive = bool(re.search(r"\b(?:bullish|outperform|overweight|buy rating|rated? buy|rating (?:of |at )?buy)\b", context, re.I)) or action in ("upgrade", "raise_target")
        negative = bool(re.search(r"\b(?:bearish|underperform|underweight|sell rating|rated? sell|rating (?:of |at )?sell)\b", context, re.I)) or action in ("downgrade", "lower_target")
        stance = "mixed" if positive and negative else "bullish" if positive else "bearish" if negative else "unknown"
        if _NEGATED_STANCE.search(context):
            stance = "unknown"
            limitations.append("Negated stance wording is left unclassified by these rules.")
    claim_kind = "unknown" if not context or uncertain or kind == "other" else "opinion" if kind in ("analyst_revision", "market_commentary") else "reported_fact"
    templates = {
        "analyst_revision": "Analyst rating or price-target commentary is reported; it is not a return forecast.",
        "partnership": "A partnership or collaboration is reported.",
        "product": "A product announcement is reported.",
        "financial_results": "Financial results or management guidance are discussed.",
        "contract_change": "A contract or commercial arrangement is discussed.",
        "market_commentary": "Market or sector commentary mentions the company.",
        "trading_activity": "Share-price movement or trading activity is reported; it is not a company operating result.",
        "other": "The company is mentioned, but the rules cannot identify a specific event.",
    }
    summary = templates[kind]
    if kind == "analyst_revision":
        summary = _analyst_summary(context)
    if relevance in ("tag_only", "none"):
        summary = "Only a vendor symbol tag links this article to the company." if tagged else "No explicit company reference was found."
        limitations.append("The article's overall tone or events are not assigned to this company.")
    if action == "termination_or_nonrenewal":
        summary = "A contract termination or non-renewal is reported; this is not a new contract award."
        limitations.append("Non-renewal and a new contract award are different events.")
    if _CREDITS.search(context):
        summary = "A cloud-credit arrangement is mentioned; credit amounts are not booked revenue."
        limitations.append("Cloud or compute credits must not be interpreted as recognized revenue.")
    if uncertain:
        summary = "Unconfirmed or conditional wording: " + summary
        limitations.append("Conditional, attributed, or denial wording requires review before treating a claim as confirmed.")
    if kind == "analyst_revision":
        limitations.append("Analyst price targets and ratings are opinions, not future realized returns.")
    if relevance == "body":
        limitations.append("The company appears in the body and may be secondary to the headline event.")
    if "<" in context and ">" in context:
        limitations.append("Markup is retained; evidence offsets refer to the original input string.")
    if not _time(article.get("created_at")):
        limitations.append("Published time is missing, invalid, or lacks a timezone; cross-article clustering is disabled.")
    tokens, brokers, numbers, action = _metadata(context, symbol, kind)
    day = _time(article.get("created_at"))
    identity = "|".join([symbol, kind, _report_day(day).isoformat() if day else "unknown", action,
                         ",".join(brokers), ",".join(numbers), ",".join(sorted(tokens))])
    if not day or relevance in ("tag_only", "none"):
        identity += "|" + str(article.get("id", "")) + "|" + context
    return {
        "article_id": article.get("id"), "symbol": symbol, "relevance": relevance,
        "event_type": kind, "claim_kind": claim_kind, "summary": f"{symbol}: {summary}",
        "evidence": evidence, "limitations": limitations,
        "event_key": f"{symbol}:{kind}:" + hashlib.sha256(identity.encode()).hexdigest()[:16],
        "stance": stance, "extraction_method": "rules_v1",
        "published_at": article.get("created_at"), "updated_at": article.get("updated_at"),
        "retrieved_at": article.get("retrieved_at"),
    }


def _same_event(left, right):
    if left.get("symbol") != right.get("symbol") or left.get("event_type") != right.get("event_type"):
        return False
    if left.get("article_id") is not None and left.get("article_id") == right.get("article_id"):
        return True
    if left.get("relevance") in ("tag_only", "none") or right.get("relevance") in ("tag_only", "none"):
        return False
    lt, rt = _time(left.get("published_at")), _time(right.get("published_at"))
    if not lt or not rt or abs((lt - rt).total_seconds()) > 3 * 86400:
        return False
    kind, symbol = left["event_type"], left["symbol"]
    if kind in ("analyst_revision", "trading_activity", "market_commentary", "other") and _report_day(lt) != _report_day(rt):
        return False
    ltext = " ".join(item["text"] for item in left.get("evidence", []))
    rtext = " ".join(item["text"] for item in right.get("evidence", []))
    ltok, lb, ln, la = _metadata(ltext, symbol, kind)
    rtok, rb, rn, ra = _metadata(rtext, symbol, kind)
    if kind == "analyst_revision":
        lb, la, lr, lp = _analyst_details(ltext)
        rb, ra, rr, rp = _analyst_details(rtext)
        if len(lb) != 1 or lb != rb or la != ra or la == "analyst_comment":
            return False
        if (lr and rr and lr != rr) or (lp and rp and lp != rp):
            return False
        # Same broker, action, local day, and explicit matching target/rating.
        # Missing details cannot contradict known ones; ambiguity is checked by
        # the caller rather than arbitrarily choosing between existing clusters.
        return bool((lr and lr == rr) or (lp and lp == rp))
    if la != ra or ln != rn or left.get("stance") != right.get("stance"):
        return False
    shared = ltok & rtok
    # Require substantive shared tokens, not just a company and event category.
    return len(shared) >= 3 and len(shared) / max(1, len(ltok | rtok)) >= 0.6


def cluster_events(extractions: list) -> list:
    """Greedy chronological clusters, comparing each record only with prior data.

    This is conservative near-duplicate grouping, not semantic event identity.
    Missing-time records are last and are not merged across article IDs. Dates
    delimit analyst/trading stories; the three-day window caps other grouping.
    """
    far_future = datetime.max.replace(tzinfo=timezone.utc)
    ordered = sorted(extractions, key=lambda row: (_time(row.get("published_at")) or far_future,
                                                  _time(row.get("updated_at")) or far_future,
                                                  str(row.get("article_id", "")), row.get("symbol", ""),
                                                  row.get("event_key", "")))
    clusters = []
    for item in ordered:
        # Every prior member must agree: an early report with a missing target
        # cannot bridge two later reports whose explicit targets conflict.
        matches = [c for c in clusters if all(_same_event(prior, item) for prior in c["extractions"])]
        cluster = matches[0] if len(matches) == 1 else None
        if cluster is None:
            cluster = {"event_key": item["event_key"], "symbol": item["symbol"], "article_ids": [],
                       "cluster_status": "candidate", "clustering_method": "rules_v1",
                       "earliest_reported_at": None, "latest_updated_at": None, "extractions": []}
            clusters.append(cluster)
        cluster["extractions"].append(item)
        if item.get("article_id") not in cluster["article_ids"]:
            cluster["article_ids"].append(item.get("article_id"))
        for source, destination, choose in (("published_at", "earliest_reported_at", min),
                                            ("updated_at", "latest_updated_at", max)):
            stamp, current = _time(item.get(source)), _time(cluster[destination])
            if stamp and (not current or choose(stamp, current) == stamp):
                cluster[destination] = item[source]
    return clusters
