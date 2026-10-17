"""Seeded synthetic IELTS Task 2 essays: the ground truth for every number in this repo.

How one essay is made:

1. A writer gets a hidden ability and, from it, a hidden quality per criterion (TR, CC, LR,
   GRA) on the band scale. The writer's first language (L1) is drawn independently of ability.
2. The essay text is assembled from templates and sentence banks. Each criterion's quality
   steers different choices:
   - TR: a clear position, a conclusion, length, and fewer off-topic sentences;
   - CC: paragraphing, topic sentences, and varied (not repeated) linking words;
   - LR: basic / mid / advanced word choices and academic collocations;
   - GRA: complex sentences, and fewer planted grammar errors.
3. **Planted L1 pattern:** writers whose L1 has no articles (Vietnamese, Finnish) drop
   "a/an/the" much more often. Ability does not depend on L1, and the raters do not see
   this pattern, so any score gap it causes is model bias, not a real quality difference.
4. Two simulated raters score each criterion: hidden quality + a prompt effect (some prompts
   are read more strictly; the text cannot show it) + a small personal bias (rater 1 slightly
   lenient, rater 2 slightly strict) + noise, rounded to a half band in 4.0-9.0.

Everything is seeded: the same seed gives byte-identical essays and scores.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import CRITERIA
from .bands import MAX_BAND, MIN_BAND, round_half_band

L1_GROUPS: tuple[str, ...] = ("Vietnamese", "Finnish", "Arabic", "Spanish")
# Probability of dropping each article. Vietnamese and Finnish have no articles.
ARTICLE_DROP: dict[str, float] = {
    "Vietnamese": 0.50,
    "Finnish": 0.45,
    "Arabic": 0.05,
    "Spanish": 0.04,
}


@dataclass(frozen=True)
class Prompt:
    prompt_id: str
    text: str
    subject: str  # a singular noun phrase used inside sentences
    group: str  # the people most affected, plural
    pros: tuple[str, ...]
    cons: tuple[str, ...]
    example: str


PROMPTS: tuple[Prompt, ...] = (
    Prompt(
        "P01",
        "Some people believe that university education should be free for all students. "
        "To what extent do you agree or disagree?",
        "free university education",
        "students",
        (
            "students from poor families get the same chance as rich students",
            "a country with more graduates usually has a stronger economy",
            "young people can choose a subject they love instead of a subject that pays",
        ),
        (
            "taxpayers have to pay for the tuition of every student",
            "some students may not take their studies seriously when they pay nothing",
            "universities may become crowded and the quality of teaching can fall",
        ),
        "in Finland students do not pay tuition fees and most young people finish a degree",
    ),
    Prompt(
        "P02",
        "Children today spend too much time on smartphones and tablets. "
        "What problems does this cause, and what can parents do about it?",
        "screen time for children",
        "children",
        (
            "children can find educational apps and videos that support their learning",
            "tablets help children stay in contact with friends and family",
            "digital skills are needed for almost every job in the future",
        ),
        (
            "children who use smartphones late at night often sleep badly",
            "too much screen time means less time for sport and outdoor games",
            "young children may see violent or unsuitable content online",
        ),
        "many parents in my city now keep phones out of the bedroom after eight o'clock",
    ),
    Prompt(
        "P03",
        "Some people say that only governments can solve climate change, while others think "
        "individuals must also act. Discuss both views and give your own opinion.",
        "action on climate change",
        "citizens",
        (
            "governments can pass laws that cut emissions from factories and cars",
            "small changes by millions of individuals add up to a large effect",
            "public investment in clean energy makes green choices cheaper for everyone",
        ),
        (
            "individuals cannot build railways or wind farms on their own",
            "some citizens feel that their personal effort makes no difference",
            "strict climate laws can raise prices for poor families",
        ),
        "Denmark produces a large share of its electricity from wind power",
    ),
    Prompt(
        "P04",
        "Many countries are introducing a tax on sugary drinks to improve public health. "
        "Do the advantages of this outweigh the disadvantages?",
        "a tax on sugary drinks",
        "consumers",
        (
            "higher prices make people buy fewer sugary drinks",
            "the money from the tax can pay for health programmes in schools",
            "companies start to make healthier drinks with less sugar",
        ),
        (
            "the tax takes a larger share of income from poor consumers",
            "people may simply switch to sweets or other unhealthy food",
            "small shops can lose customers to shops across the border",
        ),
        "after Mexico introduced a sugar tax, sales of sweet drinks fell",
    ),
    Prompt(
        "P05",
        "More and more employees now work from home. Is this a positive or negative development?",
        "working from home",
        "employees",
        (
            "employees save the time and money they used to spend on commuting",
            "parents can organise their working day around their children",
            "companies need smaller offices and can cut their costs",
        ),
        (
            "some employees feel lonely without colleagues around them",
            "it is harder for young workers to learn from experienced colleagues",
            "the line between work and private life can disappear",
        ),
        "many software companies now let staff work from home three days a week",
    ),
    Prompt(
        "P06",
        "Some people think that longer prison sentences are the best way to reduce crime. "
        "To what extent do you agree or disagree?",
        "the use of longer prison sentences",
        "offenders",
        (
            "dangerous criminals cannot harm the public while they are in prison",
            "the fear of a long sentence may stop some people from committing a crime",
            "victims may feel that justice has been done",
        ),
        (
            "prisons are expensive and the money could fund education instead",
            "many offenders learn more about crime from other prisoners",
            "former prisoners often struggle to find a job and return to crime",
        ),
        "Norway has short sentences but one of the lowest reoffending rates in Europe",
    ),
    Prompt(
        "P07",
        "Governments should spend more money on public transport than on building new roads. "
        "To what extent do you agree or disagree?",
        "investment in public transport",
        "commuters",
        (
            "buses and trains move many more commuters than private cars",
            "fewer cars on the roads means cleaner air in the city",
            "people without a car can still reach their jobs and schools",
        ),
        (
            "people in rural areas still depend on roads and cars",
            "new railway lines take many years and a lot of money to build",
            "public transport is not flexible for people with unusual working hours",
        ),
        "the new tram line in Helsinki is used by thousands of commuters every day",
    ),
    Prompt(
        "P08",
        "Nowadays many people get their news from social media rather than newspapers. "
        "What are the advantages and disadvantages of this?",
        "news on social media",
        "readers",
        (
            "readers can follow events as they happen and from many countries",
            "social media news is free and easy to access on a phone",
            "ordinary people can share stories that newspapers ignore",
        ),
        (
            "false stories can spread faster than the truth",
            "readers mostly see news that matches their own opinions",
            "local newspapers lose money and some of them close",
        ),
        "during elections, fake news stories are often shared millions of times",
    ),
    Prompt(
        "P09",
        "International tourism has grown quickly in recent decades. "
        "Do the benefits for local communities outweigh the problems?",
        "international tourism",
        "local residents",
        (
            "tourism creates jobs for local residents in hotels and restaurants",
            "visitors bring money that pays for roads and airports",
            "tourists and residents learn about each other's cultures",
        ),
        (
            "house prices and rents rise when flats are rented to tourists",
            "popular places become crowded and polluted",
            "local traditions can turn into shows for visitors",
        ),
        "the old town of Hoi An welcomes millions of tourists every year",
    ),
    Prompt(
        "P10",
        "In some countries old people are cared for by their families, in others by the state. "
        "Discuss both approaches and give your opinion.",
        "family care for older people",
        "older people",
        (
            "older people feel loved when they stay with their children and grandchildren",
            "grandparents can help with childcare and pass on family stories",
            "families save money that would go to care homes",
        ),
        (
            "caring for a parent can be a heavy burden for a working adult",
            "families often lack the medical skills that professional carers have",
            "some older people prefer not to depend on their children",
        ),
        "in Vietnam three generations often live in the same house",
    ),
    Prompt(
        "P11",
        "English is becoming a global language. Some people think this will cause many local "
        "languages to disappear. Is this a positive or negative development?",
        "the spread of English as a global language",
        "young people",
        (
            "a shared language makes international business and science easier",
            "young people who speak English can study and work abroad",
            "travellers can communicate almost everywhere",
        ),
        (
            "when a local language dies, part of the culture dies with it",
            "children may stop speaking the language of their grandparents",
            "people who do not speak English can be left behind",
        ),
        "many Finnish students read their university textbooks in English",
    ),
    Prompt(
        "P12",
        "Some people think governments should spend money on the arts, while others think "
        "the money is better spent on public services. Discuss both views.",
        "public money for the arts",
        "taxpayers",
        (
            "museums and theatres make a city attractive to visitors",
            "art helps people understand their history and identity",
            "the creative industries create many jobs",
        ),
        (
            "hospitals and schools need the money more urgently",
            "many taxpayers never visit an opera or a gallery",
            "artists can raise money from private sponsors instead",
        ),
        "the Guggenheim museum changed the image of Bilbao",
    ),
    Prompt(
        "P13",
        "Professional athletes earn far more than doctors or teachers. "
        "Is this fair? Give reasons for your answer.",
        "paying professional athletes so much",
        "athletes",
        (
            "top athletes have rare talent and very short careers",
            "sport brings huge income from tickets and television",
            "successful athletes inspire children to be active",
        ),
        (
            "doctors and teachers do work that society cannot live without",
            "huge salaries send the wrong message about what matters",
            "most of the money goes to a few stars while others earn little",
        ),
        "a single football star can earn more in a week than a nurse earns in ten years",
    ),
    Prompt(
        "P14",
        "Some people argue that money spent on space exploration should be used to solve "
        "problems on Earth. To what extent do you agree or disagree?",
        "spending money on space exploration",
        "scientists",
        (
            "space research has given us satellites, weather forecasts and GPS",
            "scientists learn more about the climate of our own planet",
            "space missions inspire young people to study science",
        ),
        (
            "millions of people still have no clean water or enough food",
            "the cost of one mission could build many schools and hospitals",
            "the benefits of space research are slow and uncertain",
        ),
        "satellite data now helps farmers plan when to plant their crops",
    ),
    Prompt(
        "P15",
        "In many cities young people cannot afford to buy a home. "
        "What are the causes of this problem and what can be done?",
        "the high cost of housing",
        "young adults",
        (
            "governments can build more affordable flats for young adults",
            "lower taxes for first-time buyers make a first home possible",
            "renting gives young adults the freedom to move for work",
        ),
        (
            "wages for young workers have not kept up with house prices",
            "investors buy flats and leave them empty",
            "there is not enough land to build in popular cities",
        ),
        "in Vienna more than half of the residents live in subsidised housing",
    ),
    Prompt(
        "P16",
        "Fast food is now widely available and cheap. Some people think advertising of fast "
        "food to children should be banned. Do you agree or disagree?",
        "a ban on fast food advertising for children",
        "parents",
        (
            "children are easily influenced by colourful adverts",
            "a ban helps parents teach their children healthy habits",
            "fewer adverts may reduce obesity in the long run",
        ),
        (
            "parents, not the government, should decide what children eat",
            "children see adverts online that a ban cannot control",
            "fast food companies may lose jobs and income",
        ),
        "the United Kingdom has limited junk food adverts on children's television",
    ),
)
PROMPTS_BY_ID: dict[str, Prompt] = {p.prompt_id: p for p in PROMPTS}

# Word banks by level for Lexical Resource: basic, mid, advanced.
LEXICON: dict[str, tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = {
    "good": (("good",), ("helpful", "useful"), ("beneficial", "advantageous", "valuable")),
    "bad": (("bad",), ("harmful", "negative"), ("detrimental", "damaging")),
    "important": (("important",), ("significant", "essential"), ("crucial", "indispensable")),
    "very": (("very",), ("really", "quite"), ("extremely", "particularly")),
    "problem": (("problem",), ("issue", "difficulty"), ("pressing concern", "major obstacle")),
    "affect": (("change",), ("influence", "affect"), ("shape", "transform")),
    "think": (("think",), ("believe", "feel"), ("would argue", "am convinced")),
    "people": (("people",), ("individuals", "citizens"), ("members of society",)),
}
COLLOCATION_UNITS: tuple[str, ...] = (
    "{Subject} can play a vital role in the modern world.",
    "{Subject} can have a profound impact on the economy.",
    "{Subject} can raise awareness of this problem among {group}.",
    "{Subject} can pose a serious threat to the way of life of {group}.",
    "{Subject} may help to bridge the gap between rich and poor.",
    "Policy makers must take into account the needs of {group}.",
    "{Subject} can affect a wide range of {people}.",
    "We need to strike a balance between the benefits and the long-term consequences.",
)
INTRO_OPENERS: tuple[str, ...] = (
    "Nowadays, many people are worried about {subject}.",
    "It has become a common topic in the modern world to discuss {subject}.",
    "{Subject} is discussed in many countries today.",
)
PERSONAL: tuple[str, ...] = (
    "In my own experience, {group} in my city often talk about {subject}.",
    "My teacher once told me that {subject} is {a_important} topic for {group}.",
    "When I discuss {subject} with my friends, we rarely agree.",
)
OFF_TOPIC: tuple[str, ...] = (
    "Many people like to travel during the summer holidays.",
    "My friend has a big family and they have a dog.",
    "The weather in my country is very hot in summer.",
    "Football is the most popular sport in the world.",
    "Life is very busy today and everyone has a phone.",
    "I like to cook traditional food with my family.",
    "Music can make people happy after a long day.",
)
CONNECTIVES_VARIED: tuple[str, ...] = (
    "Furthermore, ",
    "Moreover, ",
    "In addition, ",
    "Additionally, ",
    "As a result, ",
    "Consequently, ",
    "Similarly, ",
    "Likewise, ",
)
CONNECTIVES_BASIC: tuple[str, ...] = ("Also, ", "And ", "So ")

# Support units: lists of sentences. Simple units use short main clauses only; complex units
# add subordinate and relative clauses.
PRO_SIMPLE: tuple[tuple[str, ...], ...] = (
    ("{Subject} is {very} {good} for {group}.", "{Pro}."),
    ("{Pro}.", "This is {a_important} point."),
    ("{Pro}.", "This gives {group} more knowledge and information."),
)
PRO_COMPLEX: tuple[tuple[str, ...], ...] = (
    ("{Subject} is {very} {good} for {group} because {pro}.",),
    ("Although some people disagree, {pro}, which is {a_important} advantage.",),
    ("If the government supports {subject}, {pro}.",),
)
CON_SIMPLE: tuple[tuple[str, ...], ...] = (
    ("There is also {a_problem}.", "{Con}."),
    ("{Con}.", "This can be {very} {bad} for the country."),
    ("{Group} often say that they have no choice.", "{Con}."),
    ("{Con}.", "This does not mean that the idea is {bad}."),
)
CON_COMPLEX: tuple[tuple[str, ...], ...] = (
    ("{Subject} can be {very} {bad} because {con}.",),
    ("While {subject} has some benefits, {con}, which is {a_problem} for the government.",),
    ("If {subject} is not managed well, {con}.",),
)
# Elaborations follow a support unit; they stay on topic and use nouns that need articles.
PRO_ELABORATION: tuple[str, ...] = (
    "This is a step in the right direction for the country.",
    "In the long run, this is {a_important} benefit for {people}.",
    "It gives {group} a chance to build a better life.",
    "The government also gains from this in the long term.",
)
CON_ELABORATION: tuple[str, ...] = (
    "This is {a_problem} that the government should not ignore.",
    "For many {group}, the result is a difficult situation.",
    "It is a reason why some people are against the idea.",
    "There is no easy solution to this in the modern world.",
)
EXAMPLE_SIMPLE: tuple[str, ...] = ("For example, {example}.", "This shows that the issue is real.")
EXAMPLE_COMPLEX: tuple[str, ...] = (
    "For example, {example}, which shows that {subject} can {affect} the lives of {people}.",
)


@dataclass(frozen=True)
class GeneratorConfig:
    n_essays: int = 2000
    seed: int = 7
    ability_mean: float = 6.2
    ability_sd: float = 0.9
    criterion_sd: float = 0.45  # how much one criterion differs from the writer's ability
    rater_noise_sd: float = 0.4
    rater_bias: tuple[float, float] = (0.10, -0.15)  # rater 1 lenient, rater 2 strict
    prompt_effect_sd: dict[str, float] = field(
        default_factory=lambda: {
            "task_response": 0.35,
            "coherence_cohesion": 0.15,
            "lexical_resource": 0.15,
            "grammatical_range_accuracy": 0.15,
        }
    )
    prompts: tuple[Prompt, ...] = PROMPTS


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _a_an(phrase: str) -> str:
    return ("an " if phrase[0].lower() in "aeiou" else "a ") + phrase


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _lower_first(s: str) -> str:
    return s[:1].lower() + s[1:]


class _Writer:
    """Turns hidden qualities into text, one random choice at a time."""

    def __init__(self, rng: np.random.Generator, q: dict[str, float], l1: str, prompt: Prompt):
        self.rng, self.q, self.l1, self.p = rng, q, l1, prompt

    def _pick(self, options: tuple[str, ...] | list[str]) -> str:
        return options[int(self.rng.integers(len(options)))]

    def _word(self, slot: str) -> str:
        lr = self.q["lexical_resource"]
        p_adv, p_basic = _sigmoid((lr - 7.0) * 1.8), _sigmoid((5.8 - lr) * 1.8)
        u = self.rng.random()
        level = 2 if u < p_adv else 0 if u < p_adv + p_basic else 1
        return self._pick(LEXICON[slot][level])

    def _subject(self) -> str:
        """The full subject phrase, or 'it' (stronger writers repeat themselves less)."""
        p_ref = 0.35 * _sigmoid((self.q["lexical_resource"] - 6.5) * 1.5)
        return "it" if self.rng.random() < p_ref else self.p.subject

    def _fill(self, template: str, pro: str = "", con: str = "", full: bool = False) -> str:
        p = self.p
        subject = p.subject if full else self._subject()
        values = {
            "subject": subject,
            "Subject": _cap(subject),
            "group": p.group,
            "Group": _cap(p.group),
            "pro": pro,
            "Pro": _cap(pro),
            "con": con,
            "Con": _cap(con),
            "example": p.example,
            "very": self._word("very"),
            "good": self._word("good"),
            "bad": self._word("bad"),
            "a_important": _a_an(self._word("important")),
            "a_problem": _a_an(self._word("problem")),
            "affect": self._word("affect"),
            "think": self._word("think"),
            "people": self._word("people"),
        }
        return template.format(**values)

    def _complex(self) -> bool:
        return self.rng.random() < _sigmoid((self.q["grammatical_range_accuracy"] - 6.0) * 1.3)

    def _connective(self) -> str:
        cc = self.q["coherence_cohesion"]
        if self.rng.random() >= _sigmoid((cc - 6.0) * 1.2):
            return ""
        varied = self.rng.random() < _sigmoid((cc - 6.5) * 1.2)
        return self._pick(CONNECTIVES_VARIED if varied else CONNECTIVES_BASIC)

    def _off_topic(self) -> bool:
        p_off = float(np.clip(0.45 - 0.11 * (self.q["task_response"] - 4.5), 0.0, 0.45))
        return self.rng.random() < p_off

    def _support(self, side: str, idx: int) -> list[str]:
        if self._off_topic():
            return [self._pick(OFF_TOPIC)]
        bank = self.p.pros if side == "pro" else self.p.cons
        clause = bank[idx % len(bank)]
        if side == "pro":
            units = PRO_COMPLEX if self._complex() else PRO_SIMPLE
        else:
            units = CON_COMPLEX if self._complex() else CON_SIMPLE
        unit = units[int(self.rng.integers(len(units)))]
        sents = [self._fill(t, pro=clause, con=clause) for t in unit]
        if self.rng.random() < _sigmoid((self.q["task_response"] - 4.5) * 1.5):
            sents.append(
                self._fill(self._pick(PRO_ELABORATION if side == "pro" else CON_ELABORATION))
            )
        return sents

    def _body(self, side: str) -> list[str]:
        q = self.q
        sents: list[str] = []
        if self.rng.random() < _sigmoid((q["coherence_cohesion"] - 6.0) * 1.5):
            if side == "pro":
                sents.append(f"Firstly, {self.p.subject} has several clear advantages.")
            else:
                sents.append(f"On the other hand, {self.p.subject} also has some drawbacks.")
        n_units = 2 + int(self.rng.binomial(1, _sigmoid(q["task_response"] - 5.0)))
        start = int(self.rng.integers(3))
        for i in range(n_units):
            unit = self._support(side, start + i)
            if sents:
                conn = self._connective()
                if conn:
                    first = unit[0] if unit[0].startswith("I ") else _lower_first(unit[0])
                    unit[0] = conn + first
            sents.extend(unit)
        if side == "con" and self.rng.random() < _sigmoid((q["task_response"] - 5.0) * 1.5):
            sents.append(self._fill(self._pick(PERSONAL)))
        if side == "pro" and self.rng.random() < _sigmoid((q["task_response"] - 5.5) * 1.5):
            ex = EXAMPLE_COMPLEX if self._complex() else EXAMPLE_SIMPLE
            sents.extend(self._fill(t) for t in ex)
        if self.rng.random() < _sigmoid((q["lexical_resource"] - 6.8) * 2.0):
            sents.append(self._fill(self._pick(COLLOCATION_UNITS)))
        return sents

    def _position(self, side: int) -> str:
        tr = self.q["task_response"]
        if self.rng.random() >= _sigmoid((tr - 5.5) * 2.0):
            return self._fill("There are many good and bad things about {subject}.")
        if side > 0:
            options = (
                "In my opinion, the advantages of {subject} clearly outweigh the disadvantages.",
                "I {think} that {subject} does more good than harm.",
            )
        else:
            options = (
                "In my opinion, the disadvantages of {subject} outweigh the advantages.",
                "I {think} that {subject} does more harm than good.",
            )
        return self._fill(self._pick(options))

    def essay(self) -> str:
        q = self.q
        side = 1 if self.rng.random() < 0.5 else -1
        openers = [int(i) for i in self.rng.permutation(len(INTRO_OPENERS))]
        n_open = 2 if self.rng.random() < _sigmoid(q["task_response"] - 5.0) else 1
        intro = [self._fill(INTRO_OPENERS[i], full=True) for i in openers[:n_open]]
        intro.append(self._position(side))
        if self.rng.random() < _sigmoid((q["task_response"] - 6.0) * 1.5):
            intro.append("This essay will look at both sides of the issue.")
        bodies = [self._body("pro"), self._body("con")]
        if self.rng.random() < _sigmoid((q["task_response"] - 5.0) * 2.0):
            restated = self._position(side)
            if not restated.startswith("I "):
                restated = _lower_first(restated)
            conclusion = [
                self._fill(
                    "Although {subject} has some drawbacks, the main points above matter more."
                ),
                "In conclusion, " + restated,
            ]
        else:
            conclusion = [self._pick(OFF_TOPIC)]
        paras = [intro, *bodies, conclusion]
        paras = [[self._errors(s) for s in para] for para in paras]
        texts = [" ".join(para) for para in paras]
        if self.rng.random() >= _sigmoid((q["coherence_cohesion"] - 5.5) * 2.0):
            # Weak paragraphing: everything in one block, or an essay split at a random point.
            cut = int(self.rng.integers(1, len(texts)))
            texts = (
                [" ".join(texts)]
                if self.rng.random() < 0.5
                else [
                    " ".join(texts[:cut]),
                    " ".join(texts[cut:]),
                ]
            )
        return "\n\n".join(texts)

    def _errors(self, sentence: str) -> str:
        gra = self.q["grammatical_range_accuracy"]
        out = _drop_articles(sentence, self._article_rate(), self.rng)
        p_err = float(np.clip(0.55 - 0.12 * (gra - 4.5), 0.02, 0.6))
        if self.rng.random() < p_err:
            out = _inject_error(out, self.rng)
        return out

    def _article_rate(self) -> float:
        gra = self.q["grammatical_range_accuracy"]
        return ARTICLE_DROP[self.l1] + 0.02 * max(0.0, 6.5 - gra)


# (search, replace) pairs; each produces an error that lexicon.ERROR_PATTERNS can detect.
_ERROR_HOOKS: tuple[tuple[str, str], ...] = (
    (r"\bpeople are\b", "people is"),
    (r"\bthey have\b", "they has"),
    (r"\b(it|this|It|This) has\b", r"\1 have"),
    (r"\b(it|this|It|This) does not\b", r"\1 do not"),
    (r"\bbetter\b", "more better"),
    (r"\b(can|should|must|will) (?!to\b)", r"\1 to "),
    (r"\ban ([aeiou])", r"a \1"),
    (r"\b(information|knowledge)\b", r"\1s"),
)


def _inject_error(sentence: str, rng: np.random.Generator) -> str:
    """Apply one planted grammar error; a lower-case sentence start if nothing else fits."""
    options = [(s, r) for s, r in _ERROR_HOOKS if re.search(s, sentence)]
    if not options or rng.random() < 0.2:
        return _lower_first(sentence)
    search, replace = options[int(rng.integers(len(options)))]
    return re.sub(search, replace, sentence, count=1)


def _drop_articles(sentence: str, rate: float, rng: np.random.Generator) -> str:
    """Drop each 'a', 'an' or 'the' with probability `rate` (sentence-initial ones are kept)."""
    words = sentence.split(" ")
    kept = [
        w
        for i, w in enumerate(words)
        if not (i > 0 and w in ("a", "an", "the") and rng.random() < rate)
    ]
    return " ".join(kept)


def _rater_score(
    q: float, offset: float, bias: float, sd: float, rng: np.random.Generator
) -> float:
    raw = q + offset + bias + rng.normal(0.0, sd)
    return float(np.clip(round_half_band(raw), MIN_BAND, MAX_BAND))


def generate_essays(cfg: GeneratorConfig | None = None) -> pd.DataFrame:
    """One row per essay: text, prompt, L1, hidden qualities and both raters' bands."""
    cfg = cfg or GeneratorConfig()
    rng = np.random.default_rng(cfg.seed)
    prompt_effect = {
        p.prompt_id: {c: float(rng.normal(0.0, cfg.prompt_effect_sd[c])) for c in CRITERIA}
        for p in cfg.prompts
    }
    rows = []
    for i in range(cfg.n_essays):
        prompt = cfg.prompts[i % len(cfg.prompts)]
        l1 = L1_GROUPS[int(rng.integers(len(L1_GROUPS)))]
        ability = rng.normal(cfg.ability_mean, cfg.ability_sd)
        q = {
            c: float(np.clip(ability + rng.normal(0.0, cfg.criterion_sd), MIN_BAND, MAX_BAND))
            for c in CRITERIA
        }
        text = _Writer(rng, q, l1, prompt).essay()
        row: dict[str, object] = {
            "essay_id": f"E{i:05d}",
            "prompt_id": prompt.prompt_id,
            "prompt": prompt.text,
            "l1": l1,
            "text": text,
        }
        for c in CRITERIA:
            row[f"true_{c}"] = round(q[c], 3)
            for r, bias in enumerate(cfg.rater_bias, start=1):
                off = prompt_effect[prompt.prompt_id][c]
                row[f"r{r}_{c}"] = _rater_score(q[c], off, bias, cfg.rater_noise_sd, rng)
        rows.append(row)
    return pd.DataFrame(rows)


def human_target(df: pd.DataFrame, criterion: str) -> np.ndarray:
    """Training target: the mean of the two raters (quarter bands allowed)."""
    return ((df[f"r1_{criterion}"] + df[f"r2_{criterion}"]) / 2.0).to_numpy(dtype=float)
