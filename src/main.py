import io
import base64
import re
from fastapi import FastAPI, File, UploadFile
from fastapi.params import Query
from fastapi.responses import JSONResponse
from flask import json
from PIL import Image
from openai import OpenAI
from treys import Card, Evaluator, Deck

from src.conversation_memory import ConversationMemoryManager
from src import API_KEYS

app = FastAPI()
client = OpenAI(api_key=API_KEYS.OPENAI_API_KEY)
conversation_manager = ConversationMemoryManager()


def calculate_win_probability(player_hand, community_cards, num_players, num_simulations=1000):
    """
    Monte Carlo simulation to estimate win probability.
    player_hand: list of 2 strings, e.g. ["As", "Kd"]
    community_cards: list of 0-5 strings, e.g. ["2c", "7h", "Jd"]
    num_players: int, total number of players at the table
    """
    evaluator = Evaluator()
    player_hand_cards = [Card.new(card) for card in player_hand]
    community_cards_cards = [Card.new(card) for card in community_cards]

    wins = 0

    for _ in range(num_simulations):
        deck = Deck()
        # Remove known cards
        for card in player_hand + community_cards:
            deck.cards.remove(Card.new(card))

        # Draw opponent hands
        opponents = []
        for _ in range(num_players - 1):
            opp_hand = [deck.draw(1)[0], deck.draw(1)[0]]
            opponents.append(opp_hand)

        # Draw remaining community cards
        remaining = 5 - len(community_cards)
        sim_community = community_cards_cards + [deck.draw(1)[0] for _ in range(remaining)]

        player_score = evaluator.evaluate(sim_community, player_hand_cards)
        opp_scores = [evaluator.evaluate(sim_community, opp) for opp in opponents]

        if all(player_score <= opp_score for opp_score in opp_scores):
            wins += 1

    return wins / num_simulations


def build_user_msg(img_str: str):
    prompt = """
Based on the previous conversation for this round, and the provided picture, extract the poker state.
```json
{
  "player_hand": ["<your two cards>"],
  "community_cards": ["<list of cards on the table>"],
  "num_players": <number of players>,
  "bets": ["<list of current bets for each player>"],
  "probabilitiy_win": <probability of winning the round>,
  "action": <action to take, either "raise", "call", "fold", "check", "all-in">
}
```
"""

    return {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
        ]
    }


def analyze_poker_image(
    file: bytes,
    game_id: str = None,
    num_players: int = None
):
    if not game_id:
        return JSONResponse({"error": "game_id is required"}, status_code=400)

    img_str = base64.b64encode(file).decode()
    user_msg = build_user_msg(img_str)
    conversation_manager.set_game_id(game_id, num_players)
    conversation_manager.add_message(game_id, user_msg)

    response = client.chat.completions.create(
        model="gpt-4.1-mini-2025-04-14",
        messages=conversation_manager.get_history(game_id)
    )
    text = response.choices[0].message.content
    conversation_manager.add_message(game_id, {"role": "assistant", "content": text})

    match = re.search(r'\{.*\}', text, re.DOTALL)
    poker_state = {}
    if match:
        poker_state = json.loads(match.group(0))
    else:
        raise ValueError("Could not extract poker state from OpenAI response.")

    num_players = num_players or poker_state["num_players"]
    # Calculate odds
    win_prob = calculate_win_probability(
        poker_state["player_hand"],
        poker_state["community_cards"],
        num_players
    )

    return {
        "player_hand": poker_state["player_hand"],
        "community_cards": poker_state["community_cards"],
        "num_players": num_players,
        "win_probability_hmm": win_prob,
        "win_probability_llm": poker_state["probabilitiy_win"],
        "action": poker_state["action"],
    }


@app.post("/analyze")
def post_analyze_poker_image(
    file: UploadFile = File(...),
    game_id: str = Query(default=None),
    num_players: int = Query(default=None)
):
    image_bytes = file.file.read()
    
    return JSONResponse(analyze_poker_image(image_bytes, game_id, num_players))


if __name__ == "__main__":
    # Open image file
    image = Image.open("/Users/julienwuthrich/Downloads/toto.png")
    
    # Convert PIL Image to bytes object
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=image.format or 'PNG')
    image_bytes = img_byte_arr.getvalue()
    
    # Use the bytes object in the analyze function
    result = analyze_poker_image(image_bytes, "123", 3)
    print(result)
