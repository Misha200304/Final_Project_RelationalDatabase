from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from sqlmodel import Session, select, or_
from models import engine, Match, Player, Tournament
from typing import List, Optional
from pydantic import BaseModel
from sqlalchemy import cast, Integer, func
from fastapi import HTTPException
import torch
import torch.nn as nn


app = FastAPI()


class TennisNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


PRED_MODEL = None
PRED_FM = None
PRED_FS = None
PRED_FEATURE_COLS = None
PRED_INPUT_DIM = None


# ============================================================================
# Response Models
# ============================================================================

class PlayerMatchDetail(BaseModel):
    tourney_name: Optional[str]
    surface: Optional[str]
    round: Optional[str]
    score: Optional[str]
    opponent_name: Optional[str]
    opponent_rank: Optional[int]
    opponent_ioc: Optional[str]
    result: str  # "Win" or "Loss"
    minutes: Optional[int]
    aces: Optional[int]
    double_faults: Optional[int]
    first_serve_pct: Optional[float]
    first_serve_won_pct: Optional[float]
    second_serve_won_pct: Optional[float]
    bp_save_pct: Optional[float]


class PlayerStats(BaseModel):
    name: str
    rank: Optional[int]
    age: Optional[float]
    ioc: Optional[str]
    matches: List[PlayerMatchDetail]


class TournamentListItem(BaseModel):
    name: str
    surface: Optional[str]
    tourney_level: Optional[str]


class MatchInDraw(BaseModel):
    round: Optional[str]
    winner_name: Optional[str]
    winner_seed: Optional[int]
    winner_rank: Optional[int]
    loser_name: Optional[str]
    loser_seed: Optional[int]
    loser_rank: Optional[int]
    score: Optional[str]
    minutes: Optional[int]


class FinalistStats(BaseModel):
    aces: Optional[int]
    double_faults: Optional[int]
    first_serve_pct: Optional[float]
    first_serve_won_pct: Optional[float]
    second_serve_won_pct: Optional[float]
    bp_save_pct: Optional[float]
    total_minutes: Optional[int]


class TournamentDetail(BaseModel):
    tourney_name: str
    surface: Optional[str]
    tourney_level: str
    draw_size: Optional[int]
    tourney_date: Optional[str]
    draw: List[MatchInDraw]
    champion: Optional[str]
    finalist1_name: Optional[str]
    finalist1_stats: Optional[FinalistStats]
    finalist2_name: Optional[str]
    finalist2_stats: Optional[FinalistStats]


class PredictionResponse(BaseModel):
    player1_win_prob: float
    player2_win_prob: float
    predicted_winner: str

# ============================================================================
# Endpoints
# ============================================================================

def decode_level(code: Optional[str]) -> str:
    """Decode tourney_level code to full label."""
    level_map = {"G": "Grand Slam", "M": "Masters 1000", "A": "ATP 500/250", "F": "Tour Finals", "O": "Olympics"}
    return level_map.get(code, code or "Unknown")


def get_round_order(round_str: Optional[str]) -> int:
    """Return numeric order for sorting rounds from earliest to latest."""
    round_order = {"R128": 0, "R64": 1, "R32": 2, "R16": 3, "QF": 4, "SF": 5, "F": 6}
    return round_order.get(round_str, -1)


def load_prediction_model() -> None:
    """Load model.pth once and cache tensors/model globally."""
    global PRED_MODEL, PRED_FM, PRED_FS, PRED_FEATURE_COLS, PRED_INPUT_DIM

    model_data = torch.load("model.pth", map_location="cpu")
    PRED_FM = model_data["fm"]
    PRED_FS = model_data["fs"]
    PRED_FEATURE_COLS = model_data["feature_cols"]
    PRED_INPUT_DIM = model_data["input_dim"]

    PRED_MODEL = TennisNet(PRED_INPUT_DIM)
    PRED_MODEL.load_state_dict(model_data["parameters"])
    PRED_MODEL.eval()


@app.on_event("startup")
def startup_load_prediction_model():
    try:
        load_prediction_model()
    except Exception as e:
        # App remains usable for non-ML endpoints even if model file is absent.
        print(f"Warning: could not load model.pth at startup: {e}")


@app.get("/years")
def get_years():
    """Return distinct years from tournaments, sorted ascending."""
    with Session(engine) as session:
        # Use stored year if available, otherwise derive from tourney_id prefix (e.g. "2002-500").
        derived_year = cast(func.substr(Tournament.tourney_id, 1, 4), Integer)
        year_expr = func.coalesce(Tournament.year, derived_year)

        years = session.exec(select(year_expr).distinct()).all()
        years = sorted([int(y) for y in years if y is not None])
        return {"years": years}


@app.get("/players")
def get_players(year: int):
    """Return distinct player names who appeared in a given year, sorted by best rank (ascending)."""
    with Session(engine) as session:
        derived_year = cast(func.substr(Tournament.tourney_id, 1, 4), Integer)

        # Get all matches from tournaments in the given year
        matches = session.exec(
            select(Match).join(Tournament).where(
                or_(Tournament.year == year, derived_year == year)
            )
        ).all()
        
        # Collect each player's best (lowest) rank in the selected year.
        # Players with no available rank are placed after ranked players.
        player_best_rank = {}
        for match in matches:
            if match.winner_name:
                current = player_best_rank.get(match.winner_name)
                rank = match.winner_rank
                if current is None:
                    player_best_rank[match.winner_name] = rank
                elif rank is not None and (current is None or rank < current):
                    player_best_rank[match.winner_name] = rank

            if match.loser_name:
                current = player_best_rank.get(match.loser_name)
                rank = match.loser_rank
                if current is None:
                    player_best_rank[match.loser_name] = rank
                elif rank is not None and (current is None or rank < current):
                    player_best_rank[match.loser_name] = rank

        players = sorted(
            player_best_rank.keys(),
            key=lambda name: (
                player_best_rank[name] is None,
                player_best_rank[name] if player_best_rank[name] is not None else 10**9,
                name,
            ),
        )
        ranked_players = [
            {"name": name, "rank": player_best_rank.get(name)}
            for name in players
        ]
        return {
            "players": players,
            "ranked_players": ranked_players,
        }


@app.get("/player/{name}")
def get_player(name: str, year: int):
    """
    Return player stats for a given year and list of matches.
    Includes player rank, age, nationality from any match that year.
    """
    with Session(engine) as session:
        derived_year = cast(func.substr(Tournament.tourney_id, 1, 4), Integer)

        # Get all matches where player appears (winner or loser) in the year
        matches = session.exec(
            select(Match).join(Tournament).where(
                (or_(Tournament.year == year, derived_year == year)) &
                or_(Match.winner_name == name, Match.loser_name == name)
            )
        ).all()
        
        if not matches:
            raise HTTPException(status_code=404, detail=f"No matches found for {name} in {year}")
        
        # Extract player info from first match (rank, age, nationality)
        first_match = matches[0]
        rank = None
        age = None
        ioc = None
        
        if first_match.winner_name == name:
            rank = first_match.winner_rank
            age = first_match.winner_age
            ioc = first_match.winner_ioc
        else:
            rank = first_match.loser_rank
            age = first_match.loser_age
            ioc = first_match.loser_ioc
        
        # Build match details
        match_details = []
        for match in matches:
            is_winner = match.winner_name == name
            
            if is_winner:
                opponent_name = match.loser_name
                opponent_rank = match.loser_rank
                opponent_ioc = match.loser_ioc
                aces = match.w_ace
                dfs = match.w_df
                first_serve_pct = match.w_1st_in_pct
                first_serve_won_pct = match.w_1st_won_pct
                second_serve_won_pct = match.w_2nd_won_pct
                bp_save_pct = match.w_bp_save_pct
            else:
                opponent_name = match.winner_name
                opponent_rank = match.winner_rank
                opponent_ioc = match.winner_ioc
                aces = match.l_ace
                dfs = match.l_df
                first_serve_pct = match.l_1st_in_pct
                first_serve_won_pct = match.l_1st_won_pct
                second_serve_won_pct = match.l_2nd_won_pct
                bp_save_pct = match.l_bp_save_pct
            
            match_detail = PlayerMatchDetail(
                tourney_name=match.tournament.tourney_name if match.tournament else None,
                surface=match.tournament.surface if match.tournament else None,
                round=match.round,
                score=match.score,
                opponent_name=opponent_name,
                opponent_rank=opponent_rank,
                opponent_ioc=opponent_ioc,
                result="Win" if is_winner else "Loss",
                minutes=match.minutes,
                aces=aces,
                double_faults=dfs,
                first_serve_pct=first_serve_pct,
                first_serve_won_pct=first_serve_won_pct,
                second_serve_won_pct=second_serve_won_pct,
                bp_save_pct=bp_save_pct,
            )
            match_details.append(match_detail)
        
        return PlayerStats(
            name=name,
            rank=rank,
            age=age,
            ioc=ioc,
            matches=match_details,
        )


@app.get("/tournaments")
def get_tournaments(year: int):
    """Return distinct tournament names for a given year with surface and level."""
    with Session(engine) as session:
        derived_year = cast(func.substr(Tournament.tourney_id, 1, 4), Integer)
        
        tournaments = session.exec(
            select(Tournament).where(
                or_(Tournament.year == year, derived_year == year)
            )
        ).all()
        
        tourney_list = [
            TournamentListItem(
                name=t.tourney_name or "Unknown",
                surface=t.surface,
                tourney_level=t.tourney_level,
            )
            for t in tournaments
        ]
        
        # Sort by name
        tourney_list = sorted(tourney_list, key=lambda x: x.name)
        return {"tournaments": tourney_list}


@app.get("/tournament")
def get_tournament(name: str, year: int):
    """Get detailed tournament information including draw and finalist stats."""
    with Session(engine) as session:
        derived_year = cast(func.substr(Tournament.tourney_id, 1, 4), Integer)
        
        # Find tournament
        tournament = session.exec(
            select(Tournament).where(
                (Tournament.tourney_name == name) &
                or_(Tournament.year == year, derived_year == year)
            )
        ).first()
        
        if not tournament:
            raise HTTPException(status_code=404, detail=f"Tournament {name} not found for year {year}")
        
        # Get all matches for this tournament
        matches = session.exec(
            select(Match).where(Match.tourney_id == tournament.tourney_id)
        ).all()
        
        # Sort matches by round order
        matches_by_round = {}
        for match in matches:
            round_val = match.round or "Unknown"
            if round_val not in matches_by_round:
                matches_by_round[round_val] = []
            matches_by_round[round_val].append(match)
        
        # Sort rounds chronologically
        sorted_rounds = sorted(matches_by_round.keys(), key=get_round_order)
        
        # Build draw
        draw = []
        for round_val in sorted_rounds:
            for match in matches_by_round[round_val]:
                draw.append(MatchInDraw(
                    round=match.round,
                    winner_name=match.winner_name,
                    winner_seed=match.winner_seed,
                    winner_rank=match.winner_rank,
                    loser_name=match.loser_name,
                    loser_seed=match.loser_seed,
                    loser_rank=match.loser_rank,
                    score=match.score,
                    minutes=match.minutes,
                ))
        
        # Get final match for finalist stats and champion
        final_matches = matches_by_round.get("F", [])
        champion = None
        finalist1_name = None
        finalist1_stats = None
        finalist2_name = None
        finalist2_stats = None
        
        if final_matches:
            final_match = final_matches[0]
            champion = final_match.winner_name
            finalist1_name = final_match.winner_name
            finalist2_name = final_match.loser_name
            
            finalist1_stats = FinalistStats(
                aces=final_match.w_ace,
                double_faults=final_match.w_df,
                first_serve_pct=final_match.w_1st_in_pct,
                first_serve_won_pct=final_match.w_1st_won_pct,
                second_serve_won_pct=final_match.w_2nd_won_pct,
                bp_save_pct=final_match.w_bp_save_pct,
                total_minutes=final_match.minutes,
            )
            
            finalist2_stats = FinalistStats(
                aces=final_match.l_ace,
                double_faults=final_match.l_df,
                first_serve_pct=final_match.l_1st_in_pct,
                first_serve_won_pct=final_match.l_1st_won_pct,
                second_serve_won_pct=final_match.l_2nd_won_pct,
                bp_save_pct=final_match.l_bp_save_pct,
                total_minutes=final_match.minutes,
            )
        
        return TournamentDetail(
            tourney_name=tournament.tourney_name or "Unknown",
            surface=tournament.surface,
            tourney_level=decode_level(tournament.tourney_level),
            draw_size=tournament.draw_size,
            tourney_date=tournament.tourney_date,
            draw=draw,
            champion=champion,
            finalist1_name=finalist1_name,
            finalist1_stats=finalist1_stats,
            finalist2_name=finalist2_name,
            finalist2_stats=finalist2_stats,
        )


@app.get("/predict", response_model=PredictionResponse)
def predict_match(odds1: float, odds2: float):
    if PRED_MODEL is None or PRED_FM is None or PRED_FS is None or PRED_FEATURE_COLS is None:
        raise HTTPException(status_code=503, detail="Prediction model is not loaded. Run train.py first.")

    if odds1 <= 0 or odds2 <= 0:
        raise HTTPException(status_code=400, detail="odds1 and odds2 must be positive decimal odds values")

    # Odds-only prediction mode:
    # b365w=odds1, b365l=odds2, rank_diff=0, all one-hot surface/level columns=0.
    feature_map = {col: 0.0 for col in PRED_FEATURE_COLS}
    if "b365w" in feature_map:
        feature_map["b365w"] = float(odds1)
    if "b365l" in feature_map:
        feature_map["b365l"] = float(odds2)
    if "rank_diff" in feature_map:
        feature_map["rank_diff"] = 0.0

    features = torch.tensor(
        [[feature_map[col] for col in PRED_FEATURE_COLS]],
        dtype=torch.float32,
    )
    features = (features - PRED_FM) / PRED_FS

    with torch.no_grad():
        raw_logit = PRED_MODEL(features)
        player1_win_prob = torch.sigmoid(raw_logit).item()

    player2_win_prob = 1.0 - player1_win_prob
    predicted_winner = "Player 1" if player1_win_prob >= 0.5 else "Player 2"

    return PredictionResponse(
        player1_win_prob=player1_win_prob,
        player2_win_prob=player2_win_prob,
        predicted_winner=predicted_winner,
    )


# Mount static files
app.mount("/", StaticFiles(directory="static", html=True), name="static")