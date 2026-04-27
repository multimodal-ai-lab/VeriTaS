import random
from pathlib import Path
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from scripts.stats.common import COLORS

STOPWORDS = dict(
    es={
        "y", "o", "u", "pero", "porque", "aunque", "si", "como",
        "el", "la", "los", "las", "un", "una", "unos", "unas",
        "de", "del", "al", "a", "en", "por", "para", "con", "sin", "sobre",
        "que", "quien", "quienes", "cual", "cuales", "donde", "cuando", "como",
        "yo", "tu", "tú", "él", "ella", "nosotros", "nosotras",
        "vosotros", "vosotras", "ellos", "ellas", "se", "me", "te", "le", "les", "lo", "la",
        "mi", "mis", "tu", "tus", "su", "sus", "nuestro", "nuestra", "nuestros", "nuestras",
        "es", "son", "era", "eran", "fue", "fueron", "ser", "estar",
        "ha", "han", "haber", "hay", "había", "habían",
        "no", "nunca", "jamás", "nada", "nadie", "ningún", "ninguna",
        "más", "menos", "muy", "también", "ya", "aún", "solo", "sólo",
        "todo", "todos", "toda", "todas", "este", "esta", "estos", "estas",
        "ese", "esa", "esos", "esas", "aquel", "aquella", "aquellos", "aquellas"
    },
    hi={
        "और", "या", "लेकिन", "क्योंकि", "यदि", "तो",
        "का", "की", "के", "को", "से", "में", "पर", "लिए", "द्वारा",
        "यह", "वह", "ये", "वे", "इस", "उस", "इन", "उन",
        "मैं", "हम", "तुम", "आप", "वह", "वे",
        "मुझे", "हमें", "तुम्हें", "आपको", "उसे", "उन्हें",
        "मेरा", "मेरी", "मेरे", "हमारा", "हमारी", "हमारे",
        "तुम्हारा", "तुम्हारी", "तुम्हारे", "आपका", "आपकी", "आपके",
        "है", "हैं", "था", "थे", "थी", "हो", "होगा", "होगी",
        "नहीं", "कभी", "कभी नहीं", "कुछ", "कोई", "सब", "सभी",
        "ही", "भी", "ही नहीं", "तो भी", "ही तो",
        "बहुत", "अधिक", "कम", "पहले", "बाद", "अब"
    },
    de={
        "und", "die", "da", "zu", "nicht", "von", "hat", "den", "der", "im",
        "mit", "auf", "ein", "eine", "einer", "ist", "des", "wie", "einem",
        "das", "in", "für", "sind", "dem", "an", "bei", "dass", "es", "einen",
        "oder", "keine", "um", "sich", "aus", "beim", "al", "wurde", "als",
        "werden", "haben", "nach", "wird", "vor", "weil", "durch", "mehr",
        "zur", "unter", "am", "vom", "ihre", "alle", "zum", "auch", "bis",
        "sie", "noch", "ohne", "kann", "wurden", "ab", "gibt", "eines", "kein",
        "gab"
    },
    pt={
        "e", "ou", "mas", "porque", "como", "se", "quando", "onde",
        "o", "a", "os", "as", "um", "uma", "uns", "umas",
        "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas",
        "por", "para", "com", "sem", "sobre", "entre",
        "que", "quem", "qual", "quais", "cujo", "cuja",
        "eu", "tu", "ele", "ela", "nós", "vós", "eles", "elas",
        "me", "te", "se", "lhe", "lhes", "nos", "vos",
        "meu", "minha", "meus", "minhas", "seu", "sua", "seus", "suas",
        "é", "são", "era", "eram", "foi", "foram", "ser", "estar",
        "ter", "tem", "têm", "tinha", "tinham", "há",
        "não", "nunca", "jamais", "nada", "ninguém",
        "más", "menos", "muito", "muita", "muitos", "muitas",
        "todo", "toda", "todos", "todas", "já", "ainda"
    },
    fr={
        "et", "ou", "où", "mais", "donc", "or", "ni", "car", "qu",
        "le", "la", "les", "un", "une", "des", "du", "de", "d",
        "au", "aux", "ce", "cet", "cette", "ces",
        "il", "elle", "ils", "elles", "on", "nous", "vous", "je", "tu",
        "me", "te", "se", "lui", "leur", "y", "en",
        "mon", "ma", "mes", "ton", "ta", "tes", "son", "sa", "ses",
        "notre", "nos", "votre", "vos", "leur", "leurs",
        "est", "sont", "été", "être", "avoir", "a", "ont", "avait", "avaient",
        "sera", "seront", "été", "été", "fait", "faits", "faire",
        "ne", "pas", "plus", "jamais", "rien", "aucun", "aucune",
        "dans", "sur", "sous", "avec", "sans", "pour", "par", "contre",
        "entre", "avant", "après", "pendant", "chez",
        "que", "qui", "quoi", "dont", "où", "quand", "comment", "pourquoi",
        "si", "comme", "lorsque", "alors", "aussi", "très", "bien", "encore",
        "déjà", "tout", "tous", "toute", "toutes",
        "c", "ça", "cela", "ceci", "an"
    },
    tr={
        "ve", "veya", "ama", "çünkü", "eğer", "ise",
        "bir", "bu", "şu", "o", "şey",
        "de", "da", "den", "dan", "ile", "için", "gibi", "kadar",
        "mi", "mı", "mu", "mü", "ın", "nın",
        "ben", "sen", "o", "biz", "siz", "onlar",
        "bana", "sana", "ona", "bize", "size", "onlara",
        "benim", "senin", "onun", "bizim", "sizin", "onların",
        "var", "yok", "idi", "ise", "oldu", "olmak",
        "değil", "hiç", "her", "tüm", "çok", "daha", "en",
        "önce", "sonra", "şimdi"
    },
    ru={
        "и", "или", "но", "потому", "если", "то",
        "в", "на", "к", "с", "по", "для", "от", "о", "об", "у",
        "я", "ты", "он", "она", "оно", "мы", "вы", "они",
        "меня", "тебя", "его", "ее", "их", "нам", "вам",
        "мой", "моя", "мои", "твой", "твоя", "его", "ее", "наш", "ваш", "их",
        "это", "тот", "та", "те", "этот", "эта", "эти",
        "есть", "был", "была", "были", "быть", "будет", "будут",
        "не", "ни", "никогда", "ничего", "никто",
        "все", "всё", "весь", "вся", "всегда", "уже", "еще", "очень"
    },
    pl={
        "i", "lub", "ale", "ponieważ", "jeśli", "że", "się", "ma", "co",
        "w", "na", "do", "z", "za", "od", "o", "po", "przy", "dla",
        "ja", "ty", "on", "ona", "ono", "my", "wy", "oni", "one",
        "mnie", "ciebie", "jego", "jej", "nas", "was", "ich",
        "mój", "moja", "moje", "twój", "twoja", "twoje",
        "ten", "ta", "to", "ci", "te",
        "jest", "są", "był", "była", "byli", "być", "będzie", "będą",
        "nie", "nigdy", "nic", "nikt",
        "wszystko", "wszyscy", "zawsze", "już", "jeszcze", "bardzo"
    },
    ar={
        "و", "أو", "لكن", "لأن", "إذا", "إذ",
        "في", "على", "من", "إلى", "عن", "مع", "بدون", "بين",
        "هذا", "هذه", "هؤلاء", "ذلك", "تلك", "التي", "الذي",
        "أنا", "نحن", "أنت", "أنتم", "هو", "هي", "هم",
        "لي", "لك", "له", "لها", "لهم",
        "كان", "كانت", "يكون", "تكون", "هو", "هي",
        "لا", "لم", "لن", "ليس", "ما",
        "كل", "بعض", "كثير", "جدا", "قبل", "بعد", "الآن"
    },
)


def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    """Color function for WordCloud using colors from common.py."""
    palette = [
        COLORS["orange"],
        COLORS["soft_orange"],
        COLORS["light_orange"],
        COLORS["soft_light_orange"],
        COLORS["darkblue"],
        COLORS["blue"],
        COLORS["soft_blue"],
    ]
    return random.choice(palette)


def get_font_path(language: str) -> str | None:
    """Returns the font path for the given language if it exists."""
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf" if language != "hi" \
        else "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf"
    
    if Path(font).exists():
        return font
    return None


def plot_wordcloud_base(text: str, language: str, title: str, save: str | None = None):
    """Generates and plots a wordcloud from the given text."""
    if not text.strip():
        print(f"No text content found for {title}.")
        return

    wordcloud = WordCloud(
        font_path=get_font_path(language),
        width=1600,
        height=800,
        background_color="white",
        color_func=color_func,
        max_words=200,
        prefer_horizontal=0.7,
        min_word_length=2,
        stopwords=STOPWORDS.get(language)
    ).generate(text)

    plt.figure(figsize=(20, 10), facecolor='white')
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.title(title, fontsize=30, pad=20)

    if save:
        save_path = Path(save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved word cloud to {save_path}")

    plt.show()
