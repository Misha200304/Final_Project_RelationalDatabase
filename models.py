from typing import Optional, List
from sqlalchemy import ForeignKeyConstraint
from sqlmodel import SQLModel, Field, Relationship, create_engine, Session

DATABASE_URL = "sqlite:///tennis.db"
engine = create_engine(DATABASE_URL)


class Player(SQLModel, table=True):
    __tablename__ = "players"

    player_id: str = Field(primary_key=True)
    player_name: Optional[str] = None
    hand: Optional[str] = None
    height: Optional[int] = None
    ioc: Optional[str] = None

    matches_as_winner: List["Match"] = Relationship(
        back_populates="winner_player",
        sa_relationship_kwargs={"foreign_keys": "Match.winner_id"},
    )
    matches_as_loser: List["Match"] = Relationship(
        back_populates="loser_player",
        sa_relationship_kwargs={"foreign_keys": "Match.loser_id"},
    )
    atp_rankings: List["ATPRanking"] = Relationship(back_populates="player")


class Tournament(SQLModel, table=True):
    __tablename__ = "tournaments"

    tourney_id: str = Field(primary_key=True)
    tourney_name: Optional[str] = None
    surface: Optional[str] = None
    draw_size: Optional[int] = None
    tourney_level: Optional[str] = None
    tourney_date: Optional[str] = None
    year: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None

    matches: List["Match"] = Relationship(back_populates="tournament")


class Match(SQLModel, table=True):
    __tablename__ = "matches"

    match_id: str = Field(primary_key=True)
    tourney_id: str = Field(foreign_key="tournaments.tourney_id")
    match_num: Optional[int] = None
    winner_id: str = Field(foreign_key="players.player_id")
    winner_seed: Optional[int] = None
    winner_entry: Optional[str] = None
    winner_name: Optional[str] = None
    winner_hand: Optional[str] = None
    winner_ht: Optional[int] = None
    winner_ioc: Optional[str] = None
    winner_age: Optional[float] = None
    winner_rank: Optional[int] = None
    winner_rank_points: Optional[int] = None
    winner_odds: Optional[float] = None
    loser_id: str = Field(foreign_key="players.player_id")
    loser_seed: Optional[int] = None
    loser_entry: Optional[str] = None
    loser_name: Optional[str] = None
    loser_hand: Optional[str] = None
    loser_ht: Optional[int] = None
    loser_ioc: Optional[str] = None
    loser_age: Optional[float] = None
    loser_rank: Optional[int] = None
    loser_rank_points: Optional[int] = None
    loser_odds: Optional[float] = None
    score: Optional[str] = None
    best_of: Optional[int] = None
    round: Optional[str] = None
    minutes: Optional[int] = None
    
    # Winner statistics
    w_ace: Optional[int] = None
    w_df: Optional[int] = None
    w_svpt: Optional[int] = None
    w_1stin: Optional[int] = None
    w_1stwon: Optional[int] = None
    w_2ndwon: Optional[int] = None
    w_svgms: Optional[int] = None
    w_bpsaved: Optional[int] = None
    w_bpfaced: Optional[int] = None
    w_1st_in_pct: Optional[float] = None
    w_1st_won_pct: Optional[float] = None
    w_2nd_won_pct: Optional[float] = None
    w_ret_won: Optional[int] = None
    w_ret_won_pct: Optional[float] = None
    w_bp_conv_pct: Optional[float] = None
    w_bp_save_pct: Optional[float] = None
    w_pts: Optional[int] = None
    w_total_won_pct: Optional[float] = None
    
    # Loser statistics
    l_ace: Optional[int] = None
    l_df: Optional[int] = None
    l_svpt: Optional[int] = None
    l_1stin: Optional[int] = None
    l_1stwon: Optional[int] = None
    l_2ndwon: Optional[int] = None
    l_svgms: Optional[int] = None
    l_bpsaved: Optional[int] = None
    l_bpfaced: Optional[int] = None
    l_1st_in_pct: Optional[float] = None
    l_1st_won_pct: Optional[float] = None
    l_2nd_won_pct: Optional[float] = None
    l_ret_won: Optional[int] = None
    l_ret_won_pct: Optional[float] = None
    l_bp_conv_pct: Optional[float] = None
    l_bp_save_pct: Optional[float] = None
    l_pts: Optional[int] = None
    l_total_won_pct: Optional[float] = None
    
    # Betting odds
    b365w: Optional[float] = None
    b365l: Optional[float] = None
    
    # Probability statistics
    rank_diff: Optional[int] = None
    winner_norm: Optional[str] = None
    loser_norm: Optional[str] = None
    p_w_raw: Optional[float] = None
    p_l_raw: Optional[float] = None
    p_w: Optional[float] = None
    p_l: Optional[float] = None
    games_in_match: Optional[int] = None

    tournament: Optional[Tournament] = Relationship(back_populates="matches")
    winner_player: Optional[Player] = Relationship(
        back_populates="matches_as_winner",
        sa_relationship_kwargs={"foreign_keys": "Match.winner_id"},
    )
    loser_player: Optional[Player] = Relationship(
        back_populates="matches_as_loser",
        sa_relationship_kwargs={"foreign_keys": "Match.loser_id"},
    )


class ATPRanking(SQLModel, table=True):
    __tablename__ = "atp_rankings"

    ranking_id: int = Field(primary_key=True)
    date: str
    gender: Optional[str] = None
    rank_type: Optional[str] = Field(default=None, sa_column_kwargs={"name": "type"})
    ranking: int
    player_id: str = Field(foreign_key="players.player_id")
    country: Optional[str] = None
    age: Optional[int] = None
    points: Optional[int] = None
    tournaments: Optional[int] = None

    player: Optional[Player] = Relationship(back_populates="atp_rankings")
