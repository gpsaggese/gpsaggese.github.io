"""Pre-decision features only: no tablebase, outcomes, or future arguments."""
import re
import chess


def extract_features(context,game):
    board=chess.Board(game['start_fen'])
    for move in game['moves_uci']:board.push_uci(move)
    white=board.king(chess.WHITE);black=board.king(chess.BLACK)
    rooks=list(board.pieces(chess.ROOK,chess.WHITE));rook=rooks[0] if rooks else None
    messages=[{k:v for k,v in m.items() if k!='extra'} for m in context['messages']]
    texts=[m.get('content') or '' for m in messages]
    observations=[m.get('content') or '' for m in messages if m.get('role')=='user']
    distance=lambda a,b:chess.square_distance(a,b) if a is not None and b is not None else -1
    sf=lambda square:chess.square_file(square) if square is not None else -1
    sr=lambda square:chess.square_rank(square) if square is not None else -1
    previous_king=previous_rook=0
    if game['moves_uci']:
        prior=chess.Board(game['start_fen'])
        for i,move in enumerate(game['moves_uci']):
            if i==len(game['moves_uci'])-(2 if len(game['moves_uci'])%2==0 else 1):
                piece=prior.piece_at(chess.Move.from_uci(move).from_square)
                previous_king=int(piece is not None and piece.piece_type==chess.KING)
                previous_rook=int(piece is not None and piece.piece_type==chess.ROOK)
            prior.push_uci(move)
    return {
        'step_before':context['counters']['n_calls'],
        'remaining_agent_calls':context['counters']['step_limit']-context['counters']['n_calls'],
        'halfmove_clock':board.halfmove_clock,'halfmoves_left':100-board.halfmove_clock,
        'messages_before':len(messages),'assistant_messages_before':sum(m['role']=='assistant' for m in messages),
        'context_characters':sum(len(t) for t in texts),'context_utf8_bytes':sum(len(t.encode()) for t in texts),
        'prior_format_error_streak':context['counters']['n_consecutive_format_errors'],
        'prior_illegal_move_observations':sum('Illegal move' in t and '<returncode>1</returncode>' in t for t in observations),
        'prior_unknown_command_observations':sum('Unknown command.' in t for t in observations),
        'white_moves_so_far':(len(game['moves_uci'])+1)//2,
        'legal_moves_before':board.legal_moves.count(),'in_check_before':int(board.is_check()),
        'white_king_file':sf(white),'white_king_rank':sr(white),
        'black_king_file':sf(black),'black_king_rank':sr(black),
        'rook_file':sf(rook),'rook_rank':sr(rook),
        'king_distance':distance(white,black),'rook_black_king_distance':distance(rook,black),
        'rook_white_king_distance':distance(rook,white),
        'black_king_edge_distance':min(sf(black),7-sf(black),sr(black),7-sr(black)),
        'rook_attacked_before':int(rook is not None and board.is_attacked_by(chess.BLACK,rook)),
        'rook_defended_before':int(rook is not None and board.is_attacked_by(chess.WHITE,rook)),
        'last_white_move_was_king':previous_king,'last_white_move_was_rook':previous_rook,
    }
