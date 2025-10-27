// In src/main.rs

use clap::Parser;
use hibc_mod::api::cli::{self, Cli, Commands};

fn main() -> anyhow::Result<()> {
    // Parse the command-line arguments first.
    let cli = Cli::parse();

    // --- NEW LOGGER INITIALIZATION LOGIC ---
    // Set the log level based on the verbosity flag.
    let log_level = match cli.verbose {
        0 => log::LevelFilter::Warn,  // Default: show only warnings and errors
        1 => log::LevelFilter::Info,  // -v: show info, warnings, errors
        2 => log::LevelFilter::Debug, // -vv: show debug and lower
        _ => log::LevelFilter::Trace, // -vvv and more: show everything
    };

    env_logger::Builder::new().filter_level(log_level).init();

    // Now, dispatch to the appropriate handler function.
    match cli.command {
        Commands::Build(args) => cli::handle_build(args),
        Commands::Search(args) => cli::handle_search(args),
        Commands::Serve(args) => cli::handle_serve(args),
        Commands::Upsert(args) => cli::handle_upsert(args),
        Commands::Delete(args) => cli::handle_delete(args),
        Commands::Flush(args) => cli::handle_flush(args),
    }
}
