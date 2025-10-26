// In src/main.rs

use clap::Parser;
use hibc_mod::api::cli::{self, Cli, Commands};

fn main() -> anyhow::Result<()> {
    // Initialize a simple logger
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Parse the command-line arguments
    let cli = Cli::parse();

    // Match on the command and dispatch to the appropriate handler function
    match cli.command {
        Commands::Build(args) => cli::handle_build(args),
        Commands::Search(args) => cli::handle_search(args),
    }
}
