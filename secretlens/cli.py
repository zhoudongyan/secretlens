"""
Command Line Interface for SecretLens
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # dotenv is optional, continue without it
    pass

from .models import ScanConfig, ConfidenceLevel
from .enhanced_scanner import EnhancedScanner

console = Console()


def setup_logging(verbose: bool = False):
    """Setup logging with rich formatting"""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@click.group()
@click.version_option()
@click.pass_context
def cli(ctx):
    """
    SecretLens - LLM Enhanced Secret Detection Tool

    SecretLens combines Gitleaks' fast pattern matching with LLM semantic analysis
    to reduce false positives and discover hidden secrets in your codebase.

    Key Features:
    • Intelligent false positive filtering using LLM analysis
    • Discovery of encoded and non-standard secrets
    • Support for both Git repositories and directory scanning
    • Batch processing for multiple repositories
    • Detailed reporting with confidence scoring

    Examples:

      # Scan a single repository
      secretlens enhance ./my-repo --verbose

      # Batch scan multiple repositories
      secretlens batch-enhance repos.txt --output-dir results/

      # Generate report from previous scan
      secretlens report results.json

    For more information, visit: https://github.com/your-org/secretlens
    """
    ctx.ensure_object(dict)


@cli.command()
@click.argument("target_path", type=click.Path(exists=True))
@click.option(
    "--llm-provider",
    default="auto",
    type=click.Choice(["auto", "openai", "anthropic", "ollama"]),
    help="LLM provider to use. 'auto' detects based on available API keys, defaults to local ollama if none found.",
)
@click.option(
    "--model",
    default=os.getenv("LLM_MODEL", "gpt-4"),
    help="LLM model to use (or set LLM_MODEL environment variable)",
)
@click.option("--api-key", help="API key for LLM provider (or set via environment variable)")
@click.option(
    "--llm-base-url", help="Custom LLM API base URL (or set LLM_BASE_URL environment variable)"
)
@click.option(
    "--gitleaks-config", type=click.Path(exists=True), help="Custom Gitleaks configuration file"
)
@click.option("--output", "-o", type=click.Path(), help="Output file for results (JSON format)")
@click.option(
    "--analysis-only",
    is_flag=True,
    help="Only analyze existing Gitleaks results, skip LLM-based additional secret discovery. Faster but less comprehensive.",
)
@click.option(
    "--additional-discovery",
    is_flag=True,
    help="Enable LLM-based additional secret discovery beyond Gitleaks results. May increase API costs.",
)
@click.option(
    "--max-discovery-files",
    type=int,
    default=50,
    help="Maximum number of files to analyze for additional secrets (to control API costs).",
)
@click.option(
    "--include-low-confidence",
    is_flag=True,
    help="Include low-confidence results in output. Useful for comprehensive analysis but may increase false positives.",
)
@click.option(
    "--confidence-threshold",
    type=float,
    default=0.5,
    help="Minimum confidence threshold for results (0.0-1.0). Higher values show only high-confidence secrets, lower values include more potential matches.",
)
@click.option(
    "--scan-mode",
    type=click.Choice(["auto", "git", "dir"]),
    default="auto",
    help="Gitleaks scan mode: auto (detect), git (repository history), dir (current files)",
)
@click.option(
    "--context-window", type=int, default=2000, help="Maximum context length for LLM analysis"
)
@click.option("--batch-size", type=int, default=5, help="Batch size for parallel LLM processing")
@click.option(
    "--individual-timeout",
    type=int,
    default=120,
    help="Timeout in seconds for individual LLM analysis (default: 120s)",
)
@click.option(
    "--batch-timeout",
    type=int,
    default=600,
    help="Timeout in seconds for batch processing (default: 600s)",
)
@click.option(
    "--max-retries",
    type=int,
    default=2,
    help="Maximum retries for failed LLM analyses (default: 2)",
)
@click.option(
    "--disable-llm",
    is_flag=True,
    help="Disable LLM analysis completely, use only Gitleaks. Fastest option but no false positive filtering.",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def enhance(
    ctx,
    target_path: str,
    llm_provider: str,
    model: str,
    api_key: Optional[str],
    llm_base_url: Optional[str],
    gitleaks_config: Optional[str],
    output: Optional[str],
    analysis_only: bool,
    additional_discovery: bool,
    max_discovery_files: int,
    include_low_confidence: bool,
    confidence_threshold: float,
    scan_mode: str,
    context_window: int,
    batch_size: int,
    individual_timeout: int,
    batch_timeout: int,
    max_retries: int,
    disable_llm: bool,
    verbose: bool,
):
    """
    Enhance Gitleaks results with LLM analysis for a single target.

    This command scans a single directory or Git repository, combining Gitleaks'
    pattern matching with LLM semantic analysis to provide accurate secret detection
    with reduced false positives.

    TARGET_PATH: Path to the directory or Git repository to scan

    \b
    Workflow:
    1. Run Gitleaks to find potential secrets using regex patterns
    2. Analyze each match with LLM to determine if it's a real secret
    3. Search for additional secrets that Gitleaks might have missed
    4. Generate confidence scores and detailed analysis
    5. Display results with actionable recommendations

    \b
    Examples:

      # Basic scan with default settings
      secretlens enhance ./my-project

      # High-confidence secrets only
      secretlens enhance ./my-project --confidence-threshold 0.8

      # Include low-confidence results for comprehensive analysis
      secretlens enhance ./my-project --include-low-confidence

      # Scan Git history instead of current files
      secretlens enhance ./my-repo --scan-mode git

      # Save results to file for later analysis
      secretlens enhance ./my-project --output results.json

      # Use custom Gitleaks configuration
      secretlens enhance ./my-project --gitleaks-config custom-rules.toml

      # Enable comprehensive analysis with additional secret discovery
      secretlens enhance ./my-project --additional-discovery

      # Disable LLM analysis (Gitleaks only)
      secretlens enhance ./my-project --disable-llm

    \b
    API Key Setup:
    Create a .env file in your project directory:
      OPENAI_API_KEY=your-key-here
      ANTHROPIC_API_KEY=your-key-here
      LLM_BASE_URL=https://your-proxy.com/v1

    Or set via environment variable:
      export OPENAI_API_KEY="your-key-here"
      export ANTHROPIC_API_KEY="your-key-here"

    Or pass it directly:
      secretlens enhance ./my-project --api-key "your-key-here"

    \b
    Custom Base URL (for proxies or compatible APIs):
    Set via environment variable:
      export LLM_BASE_URL="https://your-proxy.com/v1"
      # Also supports provider-specific URLs:
      export OPENAI_BASE_URL="https://your-openai-proxy.com/v1"
      export ANTHROPIC_BASE_URL="https://your-anthropic-proxy.com"

    Or pass it directly:
      secretlens enhance ./my-project --llm-base-url "https://your-proxy.com/v1"
    """

    setup_logging(verbose)

    # Set API key from environment if not provided
    if not api_key:
        if llm_provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
        elif llm_provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")

    # Only require API key for remote providers, not for auto/ollama
    if not api_key and not disable_llm and llm_provider in ["openai", "anthropic"]:
        console.print("[red]Error: API key required for LLM analysis[/red]")
        console.print(
            f"Set {llm_provider.upper()}_API_KEY environment variable or use --api-key option"
        )
        console.print("Or use --llm-provider auto to use local AI (privacy-protected)")
        sys.exit(1)

    # Get base URLs from environment if not provided
    final_llm_base_url = llm_base_url or os.getenv("LLM_BASE_URL")

    # Create scan configuration with all Gitleaks options
    config = ScanConfig(
        target_path=target_path,
        enable_llm_analysis=not disable_llm,
        llm_provider=llm_provider,
        llm_model=model,
        max_context_length=context_window,
        batch_size=batch_size,
        include_low_confidence=include_low_confidence,
        confidence_threshold=confidence_threshold,
        verbose=verbose,
        analysis_only=analysis_only,
        enable_additional_discovery=additional_discovery,
        max_discovery_files=max_discovery_files,
        # LLM base URLs
        llm_base_url=final_llm_base_url,
        # Gitleaks-specific options
        gitleaks_config_path=gitleaks_config,
        scan_mode=scan_mode,
        gitleaks_log_level="info",
        no_banner=True,
        no_color=False,
        individual_timeout=individual_timeout,
        batch_timeout=batch_timeout,
        max_retries=max_retries,
    )

    # Store API key in environment for the scanner
    if api_key:
        if llm_provider == "openai":
            os.environ["OPENAI_API_KEY"] = api_key
        elif llm_provider == "anthropic":
            os.environ["ANTHROPIC_API_KEY"] = api_key

    # Run the scan
    console.print(Panel.fit("🔍 SecretLens Enhanced Scanning", style="bold blue"))

    scanner = EnhancedScanner(config)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:

        # Add progress task
        task = progress.add_task("Scanning...", total=None)

        try:
            # Run async scan
            result = asyncio.run(scanner.scan())

            # Update progress
            progress.update(task, description="✅ Scan completed")

        except Exception as e:
            progress.update(task, description="❌ Scan failed")
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)

    # Display results
    display_results(result)

    # Save to file if specified
    if output:
        save_results(result, output)
        console.print(f"[green]Results saved to {output}[/green]")


def display_results(result):
    """Display scan results in a formatted table"""
    console.print()

    # Summary panel
    summary = result.scan_summary
    summary_text = f"""
[bold]Files Scanned:[/bold] {result.total_files_scanned}
[bold]Total Matches:[/bold] {result.total_matches_found}
[bold]Scan Duration:[/bold] {result.scan_duration_seconds:.2f}s
[bold]Gitleaks Matches:[/bold] {summary.get('gitleaks_matches', 0)}
[bold]False Positives Filtered:[/bold] {summary.get('false_positives_filtered', 0)}
[bold]Additional Found:[/bold] {summary.get('additional_found', 0)}
[bold]False Positive Rate:[/bold] {summary.get('false_positive_rate', 0.0):.1%}
    """.strip()

    console.print(Panel(summary_text, title="📊 Scan Summary", style="green"))

    if not result.matches:
        console.print(Panel("🎉 No secrets found!", style="bold green"))
        return

    # Results table
    table = Table(title="🔐 Detected Secrets")
    table.add_column("File", style="cyan", no_wrap=True)
    table.add_column("Line", justify="right", style="magenta")
    table.add_column("Type", style="yellow")
    table.add_column("Confidence", justify="center")
    table.add_column("Method", style="blue")
    table.add_column("Status", justify="center")

    # Sort by confidence (highest first)
    sorted_matches = sorted(result.matches, key=lambda x: x.confidence, reverse=True)

    for match in sorted_matches[:20]:  # Show top 20
        # Determine status
        if match.is_likely_false_positive:
            status = "[yellow]⚠️ Likely FP[/yellow]"
        elif match.confidence >= 0.9:
            status = "[red]🚨 High Risk[/red]"
        elif match.confidence >= 0.7:
            status = "[orange1]⚡ Medium[/orange1]"
        else:
            status = "[blue]ℹ️ Low[/blue]"

        # Format confidence
        conf_str = f"{match.confidence:.2f}"
        if match.confidence >= 0.9:
            conf_str = f"[red]{conf_str}[/red]"
        elif match.confidence >= 0.7:
            conf_str = f"[orange1]{conf_str}[/orange1]"
        else:
            conf_str = f"[blue]{conf_str}[/blue]"

        # Truncate file path
        file_path = match.file_path
        if len(file_path) > 40:
            file_path = "..." + file_path[-37:]

        table.add_row(
            file_path,
            str(match.line_number),
            match.secret_type.value.replace("_", " ").title(),
            conf_str,
            match.detection_method.value.replace("_", " ").title(),
            status,
        )

    console.print(table)

    if len(result.matches) > 20:
        console.print(f"[dim]... and {len(result.matches) - 20} more matches[/dim]")

    # Show detailed analysis for high-confidence matches
    high_confidence = [
        m for m in result.matches if m.confidence >= 0.8 and not m.is_likely_false_positive
    ]
    if high_confidence:
        console.print()
        console.print(
            Panel.fit("🚨 High-Confidence Matches Require Immediate Attention", style="bold red")
        )

        for i, match in enumerate(high_confidence[:5], 1):
            details = f"""
[bold]File:[/bold] {match.file_path}:{match.line_number}
[bold]Type:[/bold] {match.secret_type.value.replace('_', ' ').title()}
[bold]Confidence:[/bold] {match.confidence:.2f}
[bold]Context:[/bold] {match.surrounding_context.strip()}
            """.strip()

            if match.llm_reasoning:
                details += f"\n[bold]Analysis:[/bold] {match.llm_reasoning}"

            console.print(Panel(details, title=f"Match #{i}", style="red"))


def save_results(result, output_path: str):
    """Save results to JSON file"""
    # Convert result to dict for JSON serialization
    result_dict = result.model_dump()

    with open(output_path, "w") as f:
        json.dump(result_dict, f, indent=2, default=str)


@cli.command()
@click.argument("results_file", type=click.Path(exists=True))
def report(results_file: str):
    """
    Generate detailed report from previous scan results.

    This command takes a JSON results file from a previous 'enhance' scan
    and generates a comprehensive human-readable report with detailed analysis,
    recommendations, and security insights.

    RESULTS_FILE: Path to the JSON results file from a previous scan

    \b
    Examples:

      # Generate report from scan results
      secretlens report results.json

      # Typical workflow
      secretlens enhance ./my-project --output scan_results.json
      secretlens report scan_results.json

    The report includes:
    • Executive summary with key metrics
    • Detailed findings grouped by confidence level
    • Security recommendations and remediation steps
    • False positive analysis and filtering insights
    """

    with open(results_file, "r") as f:
        result_dict = json.load(f)

    # Reconstruct ScanResult object
    from .models import ScanResult

    result = ScanResult(**result_dict)

    # Generate detailed report
    scanner = EnhancedScanner(result.config)
    detailed_report = scanner.generate_detailed_report(result)

    console.print(detailed_report)


@cli.command()
@click.argument("repos_file", type=click.Path(exists=True))
@click.option("--output-dir", "-o", type=click.Path(), help="Output directory for batch results")
@click.option(
    "--llm-provider",
    default="auto",
    type=click.Choice(["auto", "openai", "anthropic", "ollama"]),
    help="LLM provider to use. 'auto' detects based on available API keys, defaults to local ollama if none found.",
)
@click.option(
    "--model",
    default=os.getenv("LLM_MODEL", "gpt-4"),
    help="LLM model to use (or set LLM_MODEL environment variable)",
)
@click.option(
    "--scan-mode",
    type=click.Choice(["auto", "git", "dir"]),
    default="auto",
    help="Gitleaks scan mode: auto (detect), git (repository history), dir (current files)",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def batch_enhance(
    repos_file: str,
    output_dir: Optional[str],
    llm_provider: str,
    model: str,
    scan_mode: str,
    verbose: bool,
):
    """
    Batch process multiple repositories for enterprise-scale secret scanning.

    This command processes multiple repositories listed in a text file,
    generating individual scan results for each repository plus a consolidated
    summary report. Ideal for security audits and CI/CD integration.

    REPOS_FILE: Text file containing repository paths (one per line)

    \b
    Input File Format:
    /path/to/repo1
    /path/to/repo2
    /path/to/repo3

    \b
    Output Structure:
    output_dir/
    ├── repo1_results.json      # Individual results
    ├── repo2_results.json
    ├── repo3_results.json
    └── batch_summary.json      # Consolidated summary

    \b
    Examples:

      # Basic batch processing
      secretlens batch-enhance repos.txt

      # Specify output directory
      secretlens batch-enhance repos.txt --output-dir security_audit_2024

      # Use specific scan mode for all repositories
      secretlens batch-enhance repos.txt --scan-mode git

      # Create repository list and process
      find /projects -name ".git" -type d | sed 's|/.git||' > repos.txt
      secretlens batch-enhance repos.txt --output-dir audit_results

    \b
    Features:
    • Processes repositories sequentially with progress tracking
    • Generates individual JSON results for each repository
    • Creates consolidated summary with key metrics
    • Handles errors gracefully (skips failed repositories)
    • Uses simplified configuration for consistency across repositories

    \b
    Note: This command uses environment variables for API keys.
    Set OPENAI_API_KEY or ANTHROPIC_API_KEY before running.
    """

    setup_logging(verbose)

    # Read repository list
    with open(repos_file, "r") as f:
        repos = [line.strip() for line in f if line.strip()]

    console.print(f"[blue]Processing {len(repos)} repositories...[/blue]")

    # Create output directory
    if not output_dir:
        output_dir = f"secretlens_batch_{int(time.time())}"

    Path(output_dir).mkdir(exist_ok=True)

    results = []

    for i, repo_path in enumerate(repos, 1):
        console.print(f"[cyan]Processing {i}/{len(repos)}: {repo_path}[/cyan]")

        if not Path(repo_path).exists():
            console.print(f"[red]Warning: {repo_path} does not exist, skipping[/red]")
            continue

        try:
            # Create config for this repo
            config = ScanConfig(
                target_path=repo_path,
                llm_provider=llm_provider,
                llm_model=model,
                scan_mode=scan_mode,
                verbose=verbose,
            )

            scanner = EnhancedScanner(config)
            result = asyncio.run(scanner.scan())

            # Save individual result
            repo_name = Path(repo_path).name
            output_file = Path(output_dir) / f"{repo_name}_results.json"
            save_results(result, str(output_file))

            results.append(
                {
                    "repo": repo_path,
                    "matches": len(result.matches),
                    "high_confidence": len([m for m in result.matches if m.confidence >= 0.8]),
                    "output_file": str(output_file),
                }
            )

        except Exception as e:
            console.print(f"[red]Error processing {repo_path}: {e}[/red]")
            results.append({"repo": repo_path, "error": str(e)})

    # Generate batch summary
    summary_file = Path(output_dir) / "batch_summary.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)

    console.print(f"[green]Batch processing completed. Results in {output_dir}[/green]")


if __name__ == "__main__":
    cli()
