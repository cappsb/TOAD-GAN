from collections import OrderedDict

EMPTY_TOKEN = OrderedDict(
    {
        ".":"empty", 
    }
)

GROUND_TOKENS = OrderedDict(
    {
        "B":"ground",
        "b":"diggable", 
    }
)


SPECIAL_TOKENS = OrderedDict(
    {
        "#":"ladder",
        "G":"gold",
        "-":"rope",
    }
)

PLAYER_TOKENS = OrderedDict(
    {
        "E":"enemy"
    }
)

TOKEN_DOWNSAMPLING_HIERAECHY= [
    EMPTY_TOKEN,
    GROUND_TOKENS,
    SPECIAL_TOKENS,
    PLAYER_TOKENS
]

TOKENS = OrderedDict(
    {**EMPTY_TOKEN, **GROUND_TOKENS, **SPECIAL_TOKENS, **PLAYER_TOKENS}
)

TOKEN_GROUPS = [EMPTY_TOKEN, GROUND_TOKENS, SPECIAL_TOKENS, PLAYER_TOKENS]

REPLACE_TOKENS = {"M":"spawn"}