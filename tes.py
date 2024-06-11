from args import parse_args

if __name__ == "__main__":
    # Get args
    args = parse_args()

    if args.local_rank == 0:  # only on main process
        
        # Train model with DDP
        train(args, run)
    else:
