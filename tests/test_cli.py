from bikeflow.cli import build_parser


def test_public_commands_parse() -> None:
    parser = build_parser()
    for command in ("download", "audit", "prepare", "train", "backtest", "collect-status"):
        args = parser.parse_args([command])
        assert args.command == command
