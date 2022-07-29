from enum import Enum

states = [
    'SAME_PLAYER',
    'ADVANCE_PLAYER',
    'ACTION_INVALID',
]

envStatus = Enum('EnvStatus', states)
