from cachetools import TTLCache

default_sys_prompt = """
Extract the necessary information from a poker table image and provide the data in a structured JSON format.  

Given the image of a poker table, identify and extract the following details:
- The player's hand, which consists of two cards.
- The community cards visible on the table.
- The total number of players at the table.
- A list of current bets from each player.

Use 'null' or 'an empty list' if any information is not visible or indiscernible from the image.

# Steps

1. **Identify Player's Hand:** Locate your two cards in the image and note their representation, e.g., 'As' for Ace of spades, 'Kd' for King of diamonds.
2. **Identify Community Cards:** Determine the cards on the table, if any, and note them similarly.
3. **Count Players:** Identify the number of players present at the table.
4. **Extract Bets:** List the bets for each player as shown in the image.

# Output Format

The extracted information should be presented in the following JSON structure:

```json
{
  "player_hand": ["<your two cards>"],
  "community_cards": ["<list of cards on the table>"],
  "num_players": optional <number of players>,
  "bets": optional <list of current bets for each player>,
  "probabilitiy_win": required <probability of winning the round>,
  "action": required <action to take, either "raise", "call", "fold", "check", "all-in">
}
```

Undefined or unseen values should be represented with `null` or an empty list where appropriate.

# Notes

- Ensure that card representations are in the format of a one-letter rank followed by a one-letter suit, e.g., 'As', '7h'.
- The JSON keys and structure must be exactly as specified, with any missing data represented by `null` or `[]`.
$NUMBER_PLAYERS
"""

class ConversationMemoryManager:
    def __init__(self):
        # TTLCache with max size of 1000 items and 30 minute (1800 seconds) TTL
        self.memories = TTLCache(maxsize=1000, ttl=1800)

    def set_game_id(self, game_id: str, num_players: int = None):
        if not game_id in self.memories:
            nb_players = ""
            if num_players:
                nb_players = f"We actually know there are {num_players} players at the table."

            self.memories[game_id] = [{"role": "system", "content": default_sys_prompt.replace("$NUMBER_PLAYERS", nb_players)}]

    def get_memory(self, game_id: str) -> list[dict]:
        return self.memories[game_id]

    def add_message(self, game_id: str, message: dict):
        memory = self.get_memory(game_id)
        memory.append(message)

    def get_history(self, game_id: str) -> list[dict]:
        memory = self.get_memory(game_id)
        return memory

    def clear_history(self, game_id: str):
        if game_id in self.memories:
            del self.memories[game_id] 
