"""
Optimized database loader using SQLModel for tennis matches and ATP rankings.
Handles efficient bulk insertion with relationship management.
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Set, Optional
from sqlmodel import Session, create_engine, select, SQLModel
from models import Player, Tournament, Match, ATPRanking, engine


def load_matches_csv(csv_path: Path, session: Session) -> int:
    """
    Load tennis matches from CSV into database.
    Efficiently handles player and tournament creation.
    """
    print(f"Loading matches from {csv_path}...")
    
    players_cache: Dict[str, Player] = {}
    tournaments_cache: Dict[str, Tournament] = {}
    batch_size = 500
    match_batch = []
    inserted = 0
    skipped = 0

    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        
        for row_num, row in enumerate(reader):
            try:
                # Get or create winner player
                winner_id = row.get("winner_id", "").strip()
                if winner_id and winner_id not in players_cache:
                    existing = session.exec(
                        select(Player).where(Player.player_id == winner_id)
                    ).first()
                    if not existing:
                        winner = Player(
                            player_id=winner_id,
                            player_name=row.get("winner_name"),
                            hand=row.get("winner_hand"),
                            height=int(row["winner_ht"]) if row.get("winner_ht") and row.get("winner_ht").strip() else None,
                            ioc=row.get("winner_ioc"),
                        )
                        session.add(winner)
                        players_cache[winner_id] = winner
                    else:
                        players_cache[winner_id] = existing

                # Get or create loser player
                loser_id = row.get("loser_id", "").strip()
                if loser_id and loser_id not in players_cache:
                    existing = session.exec(
                        select(Player).where(Player.player_id == loser_id)
                    ).first()
                    if not existing:
                        loser = Player(
                            player_id=loser_id,
                            player_name=row.get("loser_name"),
                            hand=row.get("loser_hand"),
                            height=int(row["loser_ht"]) if row.get("loser_ht") and row.get("loser_ht").strip() else None,
                            ioc=row.get("loser_ioc"),
                        )
                        session.add(loser)
                        players_cache[loser_id] = loser
                    else:
                        players_cache[loser_id] = existing

                # Get or create tournament
                tourney_id = row.get("tourney_id", "").strip()
                if tourney_id and tourney_id not in tournaments_cache:
                    existing = session.exec(
                        select(Tournament).where(Tournament.tourney_id == tourney_id)
                    ).first()
                    if not existing:
                        tourney_date = row.get("tourney_date", "")
                        year = month = day = None
                        if tourney_date:
                            try:
                                dt = datetime.strptime(tourney_date, "%Y%m%d")
                                year, month, day = dt.year, dt.month, dt.day
                            except (ValueError, TypeError):
                                pass

                        tournament = Tournament(
                            tourney_id=tourney_id,
                            tourney_name=row.get("tourney_name"),
                            surface=row.get("surface"),
                            draw_size=int(row["draw_size"]) if row.get("draw_size") and row.get("draw_size").strip() else None,
                            tourney_level=row.get("tourney_level"),
                            tourney_date=tourney_date,
                            year=year,
                            month=month,
                            day=day,
                        )
                        session.add(tournament)
                        tournaments_cache[tourney_id] = tournament
                    else:
                        tournaments_cache[tourney_id] = existing

                # Create match record
                match_id = row.get("match_id", "").strip()
                if match_id:
                    match = Match(
                        match_id=match_id,
                        tourney_id=tourney_id,
                        match_num=int(row["match_num"]) if row.get("match_num") and row.get("match_num").strip() else None,
                        winner_id=winner_id,
                        winner_seed=int(row["winner_seed"]) if row.get("winner_seed") and row.get("winner_seed").strip() else None,
                        winner_entry=row.get("winner_entry"),
                        winner_name=row.get("winner_name"),
                        winner_hand=row.get("winner_hand"),
                        winner_ht=int(row["winner_ht"]) if row.get("winner_ht") and row.get("winner_ht").strip() else None,
                        winner_ioc=row.get("winner_ioc"),
                        winner_age=float(row["winner_age"]) if row.get("winner_age") and row.get("winner_age").strip() else None,
                        winner_rank=int(row["winner_rank"]) if row.get("winner_rank") and row.get("winner_rank").strip() else None,
                        winner_rank_points=int(row["winner_rank_points"]) if row.get("winner_rank_points") and row.get("winner_rank_points").strip() else None,
                        winner_odds=float(row["winner_odds"]) if row.get("winner_odds") and row.get("winner_odds").strip() else None,
                        loser_id=loser_id,
                        loser_seed=int(row["loser_seed"]) if row.get("loser_seed") and row.get("loser_seed").strip() else None,
                        loser_entry=row.get("loser_entry"),
                        loser_name=row.get("loser_name"),
                        loser_hand=row.get("loser_hand"),
                        loser_ht=int(row["loser_ht"]) if row.get("loser_ht") and row.get("loser_ht").strip() else None,
                        loser_ioc=row.get("loser_ioc"),
                        loser_age=float(row["loser_age"]) if row.get("loser_age") and row.get("loser_age").strip() else None,
                        loser_rank=int(row["loser_rank"]) if row.get("loser_rank") and row.get("loser_rank").strip() else None,
                        loser_rank_points=int(row["loser_rank_points"]) if row.get("loser_rank_points") and row.get("loser_rank_points").strip() else None,
                        loser_odds=float(row["loser_odds"]) if row.get("loser_odds") and row.get("loser_odds").strip() else None,
                        score=row.get("score"),
                        best_of=int(row["best_of"]) if row.get("best_of") and row.get("best_of").strip() else None,
                        round=row.get("round"),
                        minutes=int(row["minutes"]) if row.get("minutes") and row.get("minutes").strip() else None,
                        w_ace=int(row["w_ace"]) if row.get("w_ace") and row.get("w_ace").strip() else None,
                        w_df=int(row["w_df"]) if row.get("w_df") and row.get("w_df").strip() else None,
                        w_svpt=int(row["w_svpt"]) if row.get("w_svpt") and row.get("w_svpt").strip() else None,
                        w_1stin=int(row["w_1stin"]) if row.get("w_1stin") and row.get("w_1stin").strip() else None,
                        w_1stwon=int(row["w_1stwon"]) if row.get("w_1stwon") and row.get("w_1stwon").strip() else None,
                        w_2ndwon=int(row["w_2ndwon"]) if row.get("w_2ndwon") and row.get("w_2ndwon").strip() else None,
                        w_svgms=int(row["w_svgms"]) if row.get("w_svgms") and row.get("w_svgms").strip() else None,
                        w_bpsaved=int(row["w_bpsaved"]) if row.get("w_bpsaved") and row.get("w_bpsaved").strip() else None,
                        w_bpfaced=int(row["w_bpfaced"]) if row.get("w_bpfaced") and row.get("w_bpfaced").strip() else None,
                        w_1st_in_pct=float(row["w_1st_in_pct"]) if row.get("w_1st_in_pct") and row.get("w_1st_in_pct").strip() else None,
                        w_1st_won_pct=float(row["w_1st_won_pct"]) if row.get("w_1st_won_pct") and row.get("w_1st_won_pct").strip() else None,
                        w_2nd_won_pct=float(row["w_2nd_won_pct"]) if row.get("w_2nd_won_pct") and row.get("w_2nd_won_pct").strip() else None,
                        w_ret_won=int(row["w_ret_won"]) if row.get("w_ret_won") and row.get("w_ret_won").strip() else None,
                        w_ret_won_pct=float(row["w_ret_won_pct"]) if row.get("w_ret_won_pct") and row.get("w_ret_won_pct").strip() else None,
                        w_bp_conv_pct=float(row["w_bp_conv_pct"]) if row.get("w_bp_conv_pct") and row.get("w_bp_conv_pct").strip() else None,
                        w_bp_save_pct=float(row["w_bp_save_pct"]) if row.get("w_bp_save_pct") and row.get("w_bp_save_pct").strip() else None,
                        w_pts=int(row["w_pts"]) if row.get("w_pts") and row.get("w_pts").strip() else None,
                        w_total_won_pct=float(row["w_total_won_pct"]) if row.get("w_total_won_pct") and row.get("w_total_won_pct").strip() else None,
                        l_ace=int(row["l_ace"]) if row.get("l_ace") and row.get("l_ace").strip() else None,
                        l_df=int(row["l_df"]) if row.get("l_df") and row.get("l_df").strip() else None,
                        l_svpt=int(row["l_svpt"]) if row.get("l_svpt") and row.get("l_svpt").strip() else None,
                        l_1stin=int(row["l_1stin"]) if row.get("l_1stin") and row.get("l_1stin").strip() else None,
                        l_1stwon=int(row["l_1stwon"]) if row.get("l_1stwon") and row.get("l_1stwon").strip() else None,
                        l_2ndwon=int(row["l_2ndwon"]) if row.get("l_2ndwon") and row.get("l_2ndwon").strip() else None,
                        l_svgms=int(row["l_svgms"]) if row.get("l_svgms") and row.get("l_svgms").strip() else None,
                        l_bpsaved=int(row["l_bpsaved"]) if row.get("l_bpsaved") and row.get("l_bpsaved").strip() else None,
                        l_bpfaced=int(row["l_bpfaced"]) if row.get("l_bpfaced") and row.get("l_bpfaced").strip() else None,
                        l_1st_in_pct=float(row["l_1st_in_pct"]) if row.get("l_1st_in_pct") and row.get("l_1st_in_pct").strip() else None,
                        l_1st_won_pct=float(row["l_1st_won_pct"]) if row.get("l_1st_won_pct") and row.get("l_1st_won_pct").strip() else None,
                        l_2nd_won_pct=float(row["l_2nd_won_pct"]) if row.get("l_2nd_won_pct") and row.get("l_2nd_won_pct").strip() else None,
                        l_ret_won=int(row["l_ret_won"]) if row.get("l_ret_won") and row.get("l_ret_won").strip() else None,
                        l_ret_won_pct=float(row["l_ret_won_pct"]) if row.get("l_ret_won_pct") and row.get("l_ret_won_pct").strip() else None,
                        l_bp_conv_pct=float(row["l_bp_conv_pct"]) if row.get("l_bp_conv_pct") and row.get("l_bp_conv_pct").strip() else None,
                        l_bp_save_pct=float(row["l_bp_save_pct"]) if row.get("l_bp_save_pct") and row.get("l_bp_save_pct").strip() else None,
                        l_pts=int(row["l_pts"]) if row.get("l_pts") and row.get("l_pts").strip() else None,
                        l_total_won_pct=float(row["l_total_won_pct"]) if row.get("l_total_won_pct") and row.get("l_total_won_pct").strip() else None,
                        b365w=float(row["b365w"]) if row.get("b365w") and row.get("b365w").strip() else None,
                        b365l=float(row["b365l"]) if row.get("b365l") and row.get("b365l").strip() else None,
                        rank_diff=int(row["rank_diff"]) if row.get("rank_diff") and row.get("rank_diff").strip() else None,
                        winner_norm=row.get("winner_norm"),
                        loser_norm=row.get("loser_norm"),
                        p_w_raw=float(row["p_w_raw"]) if row.get("p_w_raw") and row.get("p_w_raw").strip() else None,
                        p_l_raw=float(row["p_l_raw"]) if row.get("p_l_raw") and row.get("p_l_raw").strip() else None,
                        p_w=float(row["p_w"]) if row.get("p_w") and row.get("p_w").strip() else None,
                        p_l=float(row["p_l"]) if row.get("p_l") and row.get("p_l").strip() else None,
                        games_in_match=int(row["games_in_match"]) if row.get("games_in_match") and row.get("games_in_match").strip() else None,
                    )
                    match_batch.append(match)

                    if len(match_batch) >= batch_size:
                        for m in match_batch:
                            session.add(m)
                        session.commit()
                        inserted += len(match_batch)
                        print(f"  Inserted {inserted} matches...")
                        match_batch.clear()

            except (ValueError, KeyError, TypeError, AttributeError) as e:
                skipped += 1
                if skipped <= 10:  # Only print first 10 errors
                    print(f"  Skipping row {row_num} due to error: {e}")
                continue

        # Insert remaining matches
        if match_batch:
            for m in match_batch:
                session.add(m)
            session.commit()
            inserted += len(match_batch)

    print(f"✓ Loaded {inserted} matches ({skipped} rows skipped)")
    return inserted


def load_atp_rankings_csv(csv_path: Path, session: Session) -> int:
    """
    Load ATP rankings from CSV into database.
    Links rankings to existing players by name matching.
    """
    print(f"Loading ATP rankings from {csv_path}...")
    
    player_name_map: Dict[str, str] = {}
    batch_size = 1000
    ranking_batch = []
    inserted = 0

    # Build player name to ID map
    for player in session.exec(select(Player)):
        if player.player_name:
            player_name_map[player.player_name] = player.player_id

    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            try:
                player_name = row.get("Player", "").strip()
                player_id = player_name_map.get(player_name)

                # Try to find player if not in map
                if not player_id:
                    existing = session.exec(
                        select(Player).where(Player.player_name == player_name)
                    ).first()
                    if existing:
                        player_id = existing.player_id
                        player_name_map[player_name] = player_id
                    else:
                        # Create new player entry if it doesn't exist
                        new_player = Player(
                            player_id=f"atp_{player_name.replace(' ', '_')}",
                            player_name=player_name,
                            ioc=row.get("Country"),
                        )
                        session.add(new_player)
                        session.flush()
                        player_id = new_player.player_id
                        player_name_map[player_name] = player_id

                # Create ranking record
                ranking = ATPRanking(
                    date=row.get("Date", ""),
                    gender=row.get("Gender"),
                    rank_type=row.get("Type"),
                    ranking=int(row["Ranking"]),
                    player_id=player_id,
                    country=row.get("Country"),
                    age=int(row["Age"]) if row.get("Age") else None,
                    points=int(row["Points"].replace(",", "")) if row.get("Points") else None,
                    tournaments=int(row["Tournaments"]) if row.get("Tournaments") else None,
                )
                ranking_batch.append(ranking)

                if len(ranking_batch) >= batch_size:
                    for r in ranking_batch:
                        session.add(r)
                    session.commit()
                    inserted += len(ranking_batch)
                    print(f"  Inserted {inserted} rankings...")
                    ranking_batch.clear()

            except (ValueError, KeyError, TypeError) as e:
                print(f"  Skipping row due to error: {e}")
                continue

        # Insert remaining rankings
        if ranking_batch:
            for r in ranking_batch:
                session.add(r)
            session.commit()
            inserted += len(ranking_batch)

    print(f"✓ Loaded {inserted} ATP rankings")
    return inserted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load tennis matches and ATP rankings into database using SQLModel."
    )
    parser.add_argument(
        "--matches-csv",
        default="correct_master.csv",
        help="Path to tennis matches CSV file.",
    )
    parser.add_argument(
        "--rankings-csv",
        default="ATP_Rankings_1990-2019.csv",
        help="Path to ATP rankings CSV file.",
    )
    parser.add_argument(
        "--db",
        default="sqlite:///tennis.db",
        help="Database URL (SQLModel format).",
    )
    parser.add_argument(
        "--skip-matches",
        action="store_true",
        help="Skip loading matches data.",
    )
    parser.add_argument(
        "--skip-rankings",
        action="store_true",
        help="Skip loading rankings data.",
    )

    args = parser.parse_args()

    # Create engine and tables
    engine_url = args.db
    db_engine = create_engine(engine_url, echo=False)
    
    print("Creating database tables...")
    SQLModel.metadata.create_all(db_engine)

    with Session(db_engine) as session:
        matches_loaded = 0
        rankings_loaded = 0

        if not args.skip_matches:
            matches_path = Path(args.matches_csv)
            if matches_path.exists():
                matches_loaded = load_matches_csv(matches_path, session)
            else:
                print(f"✗ Matches file not found: {matches_path}")

        if not args.skip_rankings:
            rankings_path = Path(args.rankings_csv)
            if rankings_path.exists():
                rankings_loaded = load_atp_rankings_csv(rankings_path, session)
            else:
                print(f"✗ Rankings file not found: {rankings_path}")

    print("\n" + "=" * 50)
    print("Database loading complete!")
    print(f"Matches:      {matches_loaded}")
    print(f"Rankings:     {rankings_loaded}")
    print(f"Database URL: {engine_url}")
    print("=" * 50)


if __name__ == "__main__":
    main()
