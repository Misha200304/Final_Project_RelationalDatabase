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
PRED_OPTIMIZER = None
PRED_CRITERION = None


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


class OnlineTrainRequest(BaseModel):
    p1_imp: float
    p2_imp: float
    rank_diff: float
    surface: str
    tourney_level: str
    p1_ace: float
    p2_ace: float
    p1_df: float
    p2_df: float
    p1_1stin: float
    p2_1stin: float
    p1_1stwon: float
    p2_1stwon: float
    p1_bpsaved: float
    p2_bpsaved: float
    p1_bpfaced: float
    p2_bpfaced: float
    p1_bp_conv_pct: float
    p2_bp_conv_pct: float
    p1_1st_serve_return_won: float
    p2_1st_serve_return_won: float
    p1_2nd_serve_return_won: float
    p2_2nd_serve_return_won: float
    target: float


class OnlineTrainResponse(BaseModel):
    p1_prob: float
    p2_prob: float
    predicted_winner: str
    loss: float
    is_correct: bool


class RandomMatchResponse(BaseModel):
    match_id: str
    tourney_name: str
    player1_name: str
    player2_name: str
    player1_rank: Optional[int]
    player2_rank: Optional[int]
    features: OnlineTrainRequest


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
    global PRED_MODEL, PRED_FM, PRED_FS, PRED_FEATURE_COLS, PRED_INPUT_DIM, PRED_OPTIMIZER, PRED_CRITERION

    model_data = torch.load("model.pth", map_location="cpu")
    PRED_FM = model_data["fm"]
    PRED_FS = model_data["fs"]
    PRED_FEATURE_COLS = model_data["feature_cols"]
    PRED_INPUT_DIM = model_data["input_dim"]

    PRED_MODEL = TennisNet(PRED_INPUT_DIM)
    PRED_MODEL.load_state_dict(model_data["parameters"])
    PRED_MODEL.eval()

    PRED_OPTIMIZER = torch.optim.Adam(PRED_MODEL.parameters(), lr=0.001)
    PRED_CRITERION = nn.BCEWithLogitsLoss()


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


import random

@app.get("/predict", response_model=PredictionResponse)
def predict_match(odds1: float, odds2: float):
    if PRED_MODEL is None or PRED_FM is None or PRED_FS is None or PRED_FEATURE_COLS is None:
        raise HTTPException(status_code=503, detail="Prediction model is not loaded. Run train.py first.")

    if odds1 <= 0 or odds2 <= 0:
        raise HTTPException(status_code=400, detail="odds1 and odds2 must be positive decimal odds values")

    feature_map = {col: 0.0 for col in PRED_FEATURE_COLS}
    if "p1_imp" in feature_map:
        feature_map["p1_imp"] = 1.0 / float(odds1)
    if "p2_imp" in feature_map:
        feature_map["p2_imp"] = 1.0 / float(odds2)
    if "rank_diff" in feature_map:
        feature_map["rank_diff"] = 0.0

    features = torch.tensor(
        [[feature_map[col] for col in PRED_FEATURE_COLS]],
        dtype=torch.float32,
    )
    features = (features - PRED_FM) / PRED_FS

    PRED_MODEL.eval()
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


@app.get("/random_match", response_model=RandomMatchResponse)
def get_random_match():
    """Fetches a random match from the DB with fully populated features."""
    with Session(engine) as session:
        # SQLite ORDER BY RANDOM() limit 1
        result = session.exec(
            select(Match).where(
                Match.w_ace.isnot(None), Match.l_ace.isnot(None),
                Match.b365w.isnot(None), Match.b365l.isnot(None),
                Match.rank_diff.isnot(None), Match.w_1stin.isnot(None),
                Match.l_1stin.isnot(None)
            ).order_by(func.random()).limit(1)
        ).first()

        if not result:
            raise HTTPException(status_code=404, detail="No sufficient stat matches found in DB")

        t = session.exec(select(Tournament).where(Tournament.tourney_id == result.tourney_id)).first()

        match = result

        # Compute derivations
        w_1st_serve_return_won = (match.l_1stin or 0) - (match.l_1stwon or 0)
        l_1st_serve_return_won = (match.w_1stin or 0) - (match.w_1stwon or 0)
        
        w_2nd_serve_return_won = ((match.l_svpt or 0) - (match.l_df or 0) - (match.l_1stin or 0)) - (match.l_2ndwon or 0)
        l_2nd_serve_return_won = ((match.w_svpt or 0) - (match.w_df or 0) - (match.w_1stin or 0)) - (match.w_2ndwon or 0)

        # Flip coin to scramble winner
        p1_is_winner = random.choice([True, False])

        if p1_is_winner:
            p1_name = match.winner_name
            p2_name = match.loser_name
            p1_rank = match.winner_rank
            p2_rank = match.loser_rank
            target = 1.0
            
            p1_imp = 1.0 / float(match.b365w)
            p2_imp = 1.0 / float(match.b365l)
            rank_diff = float(match.rank_diff)
            
            p1_ace = float(match.w_ace)
            p2_ace = float(match.l_ace)
            p1_df = float(match.w_df)
            p2_df = float(match.l_df)
            p1_1stin = float(match.w_1stin)
            p2_1stin = float(match.l_1stin)
            p1_1stwon = float(match.w_1stwon)
            p2_1stwon = float(match.l_1stwon)
            p1_bpsaved = float(match.w_bpsaved or 0)
            p2_bpsaved = float(match.l_bpsaved or 0)
            p1_bpfaced = float(match.w_bpfaced or 0)
            p2_bpfaced = float(match.l_bpfaced or 0)
            p1_bp_conv_pct = float(match.w_bp_conv_pct or 0)
            p2_bp_conv_pct = float(match.l_bp_conv_pct or 0)
            p1_1ret = float(w_1st_serve_return_won)
            p2_1ret = float(l_1st_serve_return_won)
            p1_2ret = float(w_2nd_serve_return_won)
            p2_2ret = float(l_2nd_serve_return_won)
        else:
            p1_name = match.loser_name
            p2_name = match.winner_name
            p1_rank = match.loser_rank
            p2_rank = match.winner_rank
            target = 0.0
            
            p1_imp = 1.0 / float(match.b365l)
            p2_imp = 1.0 / float(match.b365w)
            rank_diff = -float(match.rank_diff)
            
            p1_ace = float(match.l_ace)
            p2_ace = float(match.w_ace)
            p1_df = float(match.l_df)
            p2_df = float(match.w_df)
            p1_1stin = float(match.l_1stin)
            p2_1stin = float(match.w_1stin)
            p1_1stwon = float(match.l_1stwon)
            p2_1stwon = float(match.w_1stwon)
            p1_bpsaved = float(match.l_bpsaved or 0)
            p2_bpsaved = float(match.w_bpsaved or 0)
            p1_bpfaced = float(match.l_bpfaced or 0)
            p2_bpfaced = float(match.w_bpfaced or 0)
            p1_bp_conv_pct = float(match.l_bp_conv_pct or 0)
            p2_bp_conv_pct = float(match.w_bp_conv_pct or 0)
            p1_1ret = float(l_1st_serve_return_won)
            p2_1ret = float(w_1st_serve_return_won)
            p1_2ret = float(l_2nd_serve_return_won)
            p2_2ret = float(w_2nd_serve_return_won)

        req = OnlineTrainRequest(
            p1_imp=p1_imp, p2_imp=p2_imp, rank_diff=rank_diff,
            surface=t.surface or "Hard", tourney_level=t.tourney_level or "A",
            p1_ace=p1_ace, p2_ace=p2_ace, p1_df=p1_df, p2_df=p2_df,
            p1_1stin=p1_1stin, p2_1stin=p2_1stin, p1_1stwon=p1_1stwon, p2_1stwon=p2_1stwon,
            p1_bpsaved=p1_bpsaved, p2_bpsaved=p2_bpsaved,
            p1_bpfaced=p1_bpfaced, p2_bpfaced=p2_bpfaced,
            p1_bp_conv_pct=p1_bp_conv_pct, p2_bp_conv_pct=p2_bp_conv_pct,
            p1_1st_serve_return_won=p1_1ret, p2_1st_serve_return_won=p2_1ret,
            p1_2nd_serve_return_won=p1_2ret, p2_2nd_serve_return_won=p2_2ret,
            target=target
        )

        return RandomMatchResponse(
            match_id=match.match_id,
            tourney_name=t.tourney_name or "Unknown",
            player1_name=p1_name or "Unknown",
            player2_name=p2_name or "Unknown",
            player1_rank=p1_rank,
            player2_rank=p2_rank,
            features=req
        )


@app.post("/train_online", response_model=OnlineTrainResponse)
def train_online(req: OnlineTrainRequest):
    if PRED_MODEL is None or PRED_CRITERION is None or PRED_OPTIMIZER is None:
        raise HTTPException(status_code=503, detail="Prediction model is not loaded.")

    feature_map = {col: 0.0 for col in PRED_FEATURE_COLS}
    
    # Map request values generically
    for k, v in req.dict().items():
        if k in feature_map:
            feature_map[k] = float(v)
            
    # Surface and Level one-hots
    sf_key = f"surface_{req.surface}"
    lvl_key = f"tourney_level_{req.tourney_level}"
    if sf_key in feature_map: feature_map[sf_key] = 1.0
    if lvl_key in feature_map: feature_map[lvl_key] = 1.0

    # Build input tensor using EXACT column order from saved model
    features = torch.tensor([[feature_map[col] for col in PRED_FEATURE_COLS]], dtype=torch.float32)
    features = (features - PRED_FM) / PRED_FS

    target = torch.tensor([[req.target]], dtype=torch.float32)

    # 1. Forward Pass (in train mode to allow grads and dropout)
    PRED_MODEL.train()
    PRED_OPTIMIZER.zero_grad()
    raw_logit = PRED_MODEL(features)
    loss = PRED_CRITERION(raw_logit, target)
    
    # 2. Backward Pass & Step
    loss.backward()
    PRED_OPTIMIZER.step()

    # 3. Calculate Prediction (we'll just use the raw_logit we already got, though we could re-evaluate)
    with torch.no_grad():
        PRED_MODEL.eval()
        post_logit = PRED_MODEL(features)
        probs = torch.sigmoid(post_logit).item()
        
    p1_prob = probs
    p2_prob = 1.0 - probs
    predicted_winner = "Player 1" if p1_prob >= 0.5 else "Player 2"
    actual_winner = "Player 1" if req.target == 1.0 else "Player 2"
    is_correct = (predicted_winner == actual_winner)

    return OnlineTrainResponse(
        p1_prob=p1_prob,
        p2_prob=p2_prob,
        predicted_winner=predicted_winner,
        loss=loss.item(),
        is_correct=is_correct
    )


# Mount static files
app.mount("/", StaticFiles(directory="static", html=True), name="static")