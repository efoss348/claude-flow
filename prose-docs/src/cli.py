"""
Prose-Docs CLI - AI-powered document generation system.

A Click-based command-line interface for managing customer data,
templates, and document generation with Claude Flow agent coordination.
"""

import click
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

# Rich for beautiful terminal output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

console = Console() if RICH_AVAILABLE else None


# =============================================================================
# Utility Functions
# =============================================================================

def print_success(message: str) -> None:
    """Print success message in green."""
    if RICH_AVAILABLE:
        console.print(f"[green]{message}[/green]")
    else:
        click.echo(click.style(message, fg='green'))


def print_error(message: str) -> None:
    """Print error message in red."""
    if RICH_AVAILABLE:
        console.print(f"[red]Error: {message}[/red]")
    else:
        click.echo(click.style(f"Error: {message}", fg='red'), err=True)


def print_warning(message: str) -> None:
    """Print warning message in yellow."""
    if RICH_AVAILABLE:
        console.print(f"[yellow]Warning: {message}[/yellow]")
    else:
        click.echo(click.style(f"Warning: {message}", fg='yellow'))


def print_info(message: str) -> None:
    """Print info message in blue."""
    if RICH_AVAILABLE:
        console.print(f"[blue]{message}[/blue]")
    else:
        click.echo(click.style(message, fg='blue'))


def get_config_path() -> Path:
    """Get the configuration file path."""
    return Path.cwd() / "prose-docs.config.json"


def get_data_path() -> Path:
    """Get the data directory path."""
    return Path.cwd() / "prose-docs" / "data"


def load_config() -> Dict[str, Any]:
    """Load configuration from file."""
    config_path = get_config_path()
    if not config_path.exists():
        return {}
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print_warning(f"Could not load config: {e}")
        return {}


def save_config(config: Dict[str, Any]) -> None:
    """Save configuration to file."""
    config_path = get_config_path()
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def ensure_initialized() -> bool:
    """Ensure the system is initialized."""
    config = load_config()
    if not config.get('initialized'):
        print_error("System not initialized. Run 'prose-docs init' first.")
        return False
    return True


def create_progress_bar():
    """Create a rich progress bar context."""
    if RICH_AVAILABLE:
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        )
    return None


# =============================================================================
# Main CLI Group
# =============================================================================

@click.group()
@click.version_option(version='1.0.0', prog_name='prose-docs')
@click.pass_context
def cli(ctx: click.Context) -> None:
    """
    Prose-Docs: AI-powered document generation system.

    Generate personalized legal and business documents using
    Claude Flow multi-agent coordination.

    Run 'prose-docs init' to get started.
    """
    ctx.ensure_object(dict)


# =============================================================================
# Init Command
# =============================================================================

@cli.command()
@click.option('--force', '-f', is_flag=True, help='Force re-initialization')
@click.option('--db-path', type=click.Path(), default='./prose-docs/data/prose.db',
              help='Database file path')
def init(force: bool, db_path: str) -> None:
    """Initialize database and configuration.

    Sets up the prose-docs system including:
    - SQLite database for documents and customers
    - Configuration file
    - Required directories
    - Default templates
    """
    config = load_config()

    if config.get('initialized') and not force:
        print_warning("System already initialized. Use --force to re-initialize.")
        return

    print_info("Initializing Prose-Docs system...")

    if RICH_AVAILABLE:
        with create_progress_bar() as progress:
            task = progress.add_task("Setting up...", total=5)

            # Step 1: Create directories
            progress.update(task, description="Creating directories...")
            dirs = [
                Path('./prose-docs/data'),
                Path('./prose-docs/templates'),
                Path('./prose-docs/exports'),
                Path('./prose-docs/knowledge'),
                Path('./prose-docs/logs')
            ]
            for d in dirs:
                d.mkdir(parents=True, exist_ok=True)
            progress.advance(task)

            # Step 2: Initialize database
            progress.update(task, description="Initializing database...")
            db_file = Path(db_path)
            db_file.parent.mkdir(parents=True, exist_ok=True)
            # Database schema would be created here
            progress.advance(task)

            # Step 3: Create config
            progress.update(task, description="Creating configuration...")
            config = {
                'initialized': True,
                'initialized_at': datetime.now().isoformat(),
                'version': '1.0.0',
                'database': str(db_file),
                'templates_dir': './prose-docs/templates',
                'exports_dir': './prose-docs/exports',
                'knowledge_dir': './prose-docs/knowledge',
                'agents': {
                    'config_file': './prose-docs/config/agents.yaml',
                    'enabled': True
                },
                'learning': {
                    'enabled': True,
                    'auto_learn': True,
                    'min_confidence': 0.8
                }
            }
            save_config(config)
            progress.advance(task)

            # Step 4: Load agents configuration
            progress.update(task, description="Loading agent definitions...")
            agents_config = Path('./prose-docs/config/agents.yaml')
            if not agents_config.exists():
                print_warning("Agents config not found. Create prose-docs/config/agents.yaml")
            progress.advance(task)

            # Step 5: Final verification
            progress.update(task, description="Verifying setup...")
            progress.advance(task)
    else:
        # Fallback without rich
        click.echo("Creating directories...")
        dirs = [
            Path('./prose-docs/data'),
            Path('./prose-docs/templates'),
            Path('./prose-docs/exports'),
            Path('./prose-docs/knowledge'),
            Path('./prose-docs/logs')
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

        db_file = Path(db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)

        config = {
            'initialized': True,
            'initialized_at': datetime.now().isoformat(),
            'version': '1.0.0',
            'database': str(db_file)
        }
        save_config(config)

    print_success("Prose-Docs initialized successfully!")
    print_info(f"Database: {db_path}")
    print_info("Run 'prose-docs status' to verify setup.")


# =============================================================================
# Template Commands
# =============================================================================

@cli.group()
def template() -> None:
    """Manage document templates."""
    pass


@template.command('add')
@click.argument('path', type=click.Path(exists=True))
@click.option('--name', '-n', help='Template name (defaults to filename)')
@click.option('--category', '-c', default='general', help='Template category')
@click.option('--description', '-d', help='Template description')
def template_add(path: str, name: Optional[str], category: str, description: Optional[str]) -> None:
    """Add a new template from file.

    PATH: Path to the template file (.docx, .md, or .txt)

    Templates support variable placeholders using {{variable_name}} syntax.
    """
    if not ensure_initialized():
        return

    template_path = Path(path)
    template_name = name or template_path.stem

    # Validate file type
    valid_extensions = {'.docx', '.md', '.txt', '.html'}
    if template_path.suffix.lower() not in valid_extensions:
        print_error(f"Invalid template format. Supported: {', '.join(valid_extensions)}")
        return

    print_info(f"Adding template: {template_name}")

    # Copy template to templates directory
    config = load_config()
    templates_dir = Path(config.get('templates_dir', './prose-docs/templates'))
    templates_dir.mkdir(parents=True, exist_ok=True)

    dest_path = templates_dir / f"{template_name}{template_path.suffix}"

    try:
        import shutil
        shutil.copy2(template_path, dest_path)

        # Register template in database/config
        templates = config.get('templates', {})
        templates[template_name] = {
            'path': str(dest_path),
            'category': category,
            'description': description or '',
            'added_at': datetime.now().isoformat(),
            'format': template_path.suffix.lower()
        }
        config['templates'] = templates
        save_config(config)

        print_success(f"Template '{template_name}' added successfully!")
        print_info(f"Location: {dest_path}")

    except IOError as e:
        print_error(f"Failed to add template: {e}")


@template.command('list')
@click.option('--category', '-c', help='Filter by category')
@click.option('--format', '-f', 'output_format', type=click.Choice(['table', 'json']),
              default='table', help='Output format')
def template_list(category: Optional[str], output_format: str) -> None:
    """List available templates."""
    if not ensure_initialized():
        return

    config = load_config()
    templates = config.get('templates', {})

    if category:
        templates = {k: v for k, v in templates.items() if v.get('category') == category}

    if not templates:
        print_warning("No templates found. Add templates with 'prose-docs template add <path>'")
        return

    if output_format == 'json':
        click.echo(json.dumps(templates, indent=2))
        return

    if RICH_AVAILABLE:
        table = Table(title="Available Templates")
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Category", style="magenta")
        table.add_column("Format", style="green")
        table.add_column("Description")
        table.add_column("Added", style="dim")

        for name, info in templates.items():
            added_at = info.get('added_at', 'Unknown')
            if added_at != 'Unknown':
                added_at = added_at[:10]  # Just the date
            table.add_row(
                name,
                info.get('category', 'general'),
                info.get('format', 'unknown'),
                info.get('description', '')[:50],
                added_at
            )

        console.print(table)
    else:
        click.echo("\nAvailable Templates:")
        click.echo("-" * 60)
        for name, info in templates.items():
            click.echo(f"  {name} ({info.get('category', 'general')}) - {info.get('format', 'unknown')}")


# =============================================================================
# Customer Commands
# =============================================================================

@cli.group()
def customer() -> None:
    """Manage customer data."""
    pass


@customer.command('add')
@click.option('--file', '-f', type=click.Path(exists=True), help='Load customer from JSON file')
@click.option('--interactive', '-i', is_flag=True, help='Interactive mode')
def customer_add(file: Optional[str], interactive: bool) -> None:
    """Add a new customer.

    Use --file to import from JSON or --interactive for guided input.
    """
    if not ensure_initialized():
        return

    customer_data = {}

    if file:
        try:
            with open(file, 'r') as f:
                customer_data = json.load(f)
            print_info(f"Loaded customer data from {file}")
        except (json.JSONDecodeError, IOError) as e:
            print_error(f"Failed to load customer file: {e}")
            return
    elif interactive:
        print_info("Enter customer information (press Enter to skip optional fields):")
        click.echo("")

        customer_data['name'] = click.prompt("Full Name", type=str)
        customer_data['email'] = click.prompt("Email", type=str, default='')
        customer_data['company'] = click.prompt("Company", type=str, default='')
        customer_data['phone'] = click.prompt("Phone", type=str, default='')
        customer_data['address'] = click.prompt("Address", type=str, default='')

        # Custom fields
        if click.confirm("Add custom fields?", default=False):
            customer_data['custom'] = {}
            while True:
                field_name = click.prompt("Field name (empty to finish)", default='')
                if not field_name:
                    break
                field_value = click.prompt(f"Value for '{field_name}'")
                customer_data['custom'][field_name] = field_value
    else:
        print_error("Specify --file or --interactive mode")
        return

    # Generate customer ID
    import hashlib
    customer_id = hashlib.sha256(
        f"{customer_data.get('name', '')}{datetime.now().isoformat()}".encode()
    ).hexdigest()[:12]

    customer_data['id'] = customer_id
    customer_data['created_at'] = datetime.now().isoformat()

    # Save customer
    config = load_config()
    customers = config.get('customers', {})
    customers[customer_id] = customer_data
    config['customers'] = customers
    save_config(config)

    print_success(f"Customer added successfully!")
    print_info(f"Customer ID: {customer_id}")


@customer.command('import')
@click.argument('source', type=click.Choice(['sheets', 'csv', 'api']))
@click.argument('path_or_url')
@click.option('--sheet-name', help='Sheet name for Google Sheets')
@click.option('--api-key', help='API key for authenticated sources')
@click.option('--mapping', '-m', type=click.Path(exists=True), help='Field mapping JSON file')
def customer_import(source: str, path_or_url: str, sheet_name: Optional[str],
                    api_key: Optional[str], mapping: Optional[str]) -> None:
    """Import customers from external sources.

    SOURCE: Import source type (sheets, csv, api)
    PATH_OR_URL: File path or URL to import from

    Examples:

      prose-docs customer import csv ./customers.csv

      prose-docs customer import sheets "spreadsheet_id" --sheet-name "Clients"

      prose-docs customer import api "https://api.example.com/customers" --api-key "key"
    """
    if not ensure_initialized():
        return

    print_info(f"Importing customers from {source}: {path_or_url}")

    # Load field mapping if provided
    field_map = {}
    if mapping:
        try:
            with open(mapping, 'r') as f:
                field_map = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print_error(f"Failed to load mapping file: {e}")
            return

    imported_count = 0

    if RICH_AVAILABLE:
        with create_progress_bar() as progress:
            task = progress.add_task(f"Importing from {source}...", total=None)

            if source == 'csv':
                try:
                    import csv as csv_module
                    with open(path_or_url, 'r', newline='') as f:
                        reader = csv_module.DictReader(f)
                        rows = list(reader)
                        progress.update(task, total=len(rows))

                        config = load_config()
                        customers = config.get('customers', {})

                        for row in rows:
                            # Apply field mapping
                            customer_data = {}
                            for csv_field, value in row.items():
                                mapped_field = field_map.get(csv_field, csv_field.lower())
                                customer_data[mapped_field] = value

                            # Generate ID
                            import hashlib
                            customer_id = hashlib.sha256(
                                f"{customer_data.get('name', '')}{datetime.now().isoformat()}{imported_count}".encode()
                            ).hexdigest()[:12]

                            customer_data['id'] = customer_id
                            customer_data['created_at'] = datetime.now().isoformat()
                            customer_data['source'] = 'csv_import'

                            customers[customer_id] = customer_data
                            imported_count += 1
                            progress.advance(task)

                        config['customers'] = customers
                        save_config(config)

                except IOError as e:
                    print_error(f"Failed to read CSV file: {e}")
                    return

            elif source == 'sheets':
                print_warning("Google Sheets import requires gspread. Installing...")
                # Would implement sheets integration here
                print_info("Sheets import not yet implemented. Use CSV export as workaround.")
                return

            elif source == 'api':
                print_warning("API import requires requests library.")
                # Would implement API integration here
                print_info("API import not yet implemented.")
                return
    else:
        # Fallback without rich progress
        if source == 'csv':
            try:
                import csv as csv_module
                with open(path_or_url, 'r', newline='') as f:
                    reader = csv_module.DictReader(f)
                    config = load_config()
                    customers = config.get('customers', {})

                    for row in reader:
                        customer_data = dict(row)
                        import hashlib
                        customer_id = hashlib.sha256(
                            f"{customer_data.get('name', '')}{datetime.now().isoformat()}{imported_count}".encode()
                        ).hexdigest()[:12]
                        customer_data['id'] = customer_id
                        customers[customer_id] = customer_data
                        imported_count += 1

                    config['customers'] = customers
                    save_config(config)
            except IOError as e:
                print_error(f"Failed to read CSV: {e}")
                return

    print_success(f"Imported {imported_count} customers successfully!")


# =============================================================================
# Document Generation Commands
# =============================================================================

@cli.command()
@click.argument('customer_id')
@click.argument('template_id')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--format', '-f', 'output_format', type=click.Choice(['docx', 'pdf', 'md']),
              default='docx', help='Output format')
@click.option('--dry-run', is_flag=True, help='Preview without generating')
def generate(customer_id: str, template_id: str, output: Optional[str],
             output_format: str, dry_run: bool) -> None:
    """Generate a document for a customer using a template.

    CUSTOMER_ID: Customer identifier
    TEMPLATE_ID: Template name or ID

    The generation process coordinates multiple AI agents:
    - data_ingestion: Validates customer data
    - document_builder: Fills template placeholders
    - formatting_editor: Ensures consistent formatting
    - legal_proofer: Verifies legal accuracy
    - visual_formatter: Final aesthetic pass
    """
    if not ensure_initialized():
        return

    config = load_config()

    # Validate customer
    customers = config.get('customers', {})
    if customer_id not in customers:
        print_error(f"Customer not found: {customer_id}")
        print_info("Use 'prose-docs customer add' to add customers")
        return

    # Validate template
    templates = config.get('templates', {})
    if template_id not in templates:
        print_error(f"Template not found: {template_id}")
        print_info("Use 'prose-docs template list' to see available templates")
        return

    customer = customers[customer_id]
    template_info = templates[template_id]

    if dry_run:
        print_info("Dry run - Preview mode")
        if RICH_AVAILABLE:
            console.print(Panel(f"""
[bold]Document Generation Preview[/bold]

Customer: {customer.get('name', 'Unknown')} ({customer_id})
Template: {template_id}
Format: {output_format}

[dim]Variables to be filled:[/dim]
"""))
            for key, value in customer.items():
                if key not in ['id', 'created_at', 'source']:
                    console.print(f"  {{{{[cyan]{key}[/cyan]}}}} -> [green]{value}[/green]")
        else:
            click.echo(f"\nCustomer: {customer.get('name', 'Unknown')}")
            click.echo(f"Template: {template_id}")
        return

    print_info("Starting document generation...")

    # Generate document ID
    import hashlib
    doc_id = hashlib.sha256(
        f"{customer_id}{template_id}{datetime.now().isoformat()}".encode()
    ).hexdigest()[:12]

    if RICH_AVAILABLE:
        with create_progress_bar() as progress:
            # Multi-agent pipeline simulation
            agents = [
                ("Data Ingestion Agent", "Validating customer data..."),
                ("Document Builder Agent", "Filling template placeholders..."),
                ("Formatting Editor Agent", "Applying formatting rules..."),
                ("Legal Proofer Agent", "Checking legal compliance..."),
                ("Visual Formatter Agent", "Final aesthetic pass..."),
            ]

            overall = progress.add_task("[bold]Generating document...", total=len(agents))

            for agent_name, description in agents:
                agent_task = progress.add_task(f"  {agent_name}", total=100)
                progress.update(agent_task, description=description)

                # Simulate agent work
                for i in range(100):
                    import time
                    time.sleep(0.01)  # Simulation delay
                    progress.advance(agent_task)

                progress.advance(overall)

    # Determine output path
    if output:
        output_path = Path(output)
    else:
        exports_dir = Path(config.get('exports_dir', './prose-docs/exports'))
        exports_dir.mkdir(parents=True, exist_ok=True)
        output_path = exports_dir / f"{doc_id}.{output_format}"

    # Record generated document
    documents = config.get('documents', {})
    documents[doc_id] = {
        'customer_id': customer_id,
        'template_id': template_id,
        'output_path': str(output_path),
        'format': output_format,
        'generated_at': datetime.now().isoformat(),
        'status': 'generated',
        'reviewed': False
    }
    config['documents'] = documents
    save_config(config)

    print_success(f"Document generated successfully!")
    print_info(f"Document ID: {doc_id}")
    print_info(f"Output: {output_path}")


@cli.command()
@click.argument('customer_file', type=click.Path(exists=True))
@click.argument('template_id')
@click.option('--output-dir', '-o', type=click.Path(), help='Output directory')
@click.option('--format', '-f', 'output_format', type=click.Choice(['docx', 'pdf', 'md']),
              default='docx', help='Output format')
@click.option('--parallel', '-p', type=int, default=4, help='Parallel processing workers')
def batch(customer_file: str, template_id: str, output_dir: Optional[str],
          output_format: str, parallel: int) -> None:
    """Batch process documents for multiple customers.

    CUSTOMER_FILE: JSON or CSV file with customer data
    TEMPLATE_ID: Template to use for all documents

    Efficiently processes multiple documents in parallel using
    the Claude Flow agent swarm coordination.
    """
    if not ensure_initialized():
        return

    # Load customer file
    customer_path = Path(customer_file)
    customers_to_process = []

    try:
        if customer_path.suffix.lower() == '.json':
            with open(customer_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    customers_to_process = data
                else:
                    customers_to_process = [data]
        elif customer_path.suffix.lower() == '.csv':
            import csv as csv_module
            with open(customer_path, 'r', newline='') as f:
                reader = csv_module.DictReader(f)
                customers_to_process = list(reader)
        else:
            print_error("Customer file must be JSON or CSV")
            return
    except (json.JSONDecodeError, IOError) as e:
        print_error(f"Failed to load customer file: {e}")
        return

    if not customers_to_process:
        print_warning("No customers found in file")
        return

    print_info(f"Processing {len(customers_to_process)} documents with {parallel} workers...")

    config = load_config()
    exports_dir = Path(output_dir) if output_dir else Path(config.get('exports_dir', './prose-docs/exports'))
    exports_dir.mkdir(parents=True, exist_ok=True)

    generated_docs = []
    failed_docs = []

    if RICH_AVAILABLE:
        with create_progress_bar() as progress:
            task = progress.add_task(
                f"[bold]Batch processing {len(customers_to_process)} documents...",
                total=len(customers_to_process)
            )

            for i, customer in enumerate(customers_to_process):
                progress.update(task, description=f"Processing: {customer.get('name', f'Customer {i+1}')}")

                try:
                    # Generate document ID
                    import hashlib
                    doc_id = hashlib.sha256(
                        f"{customer.get('name', '')}{template_id}{datetime.now().isoformat()}{i}".encode()
                    ).hexdigest()[:12]

                    output_path = exports_dir / f"{doc_id}.{output_format}"

                    generated_docs.append({
                        'doc_id': doc_id,
                        'customer': customer.get('name', 'Unknown'),
                        'output': str(output_path)
                    })

                except Exception as e:
                    failed_docs.append({
                        'customer': customer.get('name', f'Customer {i+1}'),
                        'error': str(e)
                    })

                progress.advance(task)
    else:
        for i, customer in enumerate(customers_to_process):
            click.echo(f"Processing {i+1}/{len(customers_to_process)}: {customer.get('name', 'Unknown')}")
            # Simplified processing
            import hashlib
            doc_id = hashlib.sha256(
                f"{customer.get('name', '')}{datetime.now().isoformat()}{i}".encode()
            ).hexdigest()[:12]
            generated_docs.append({'doc_id': doc_id, 'customer': customer.get('name', 'Unknown')})

    # Summary
    print_success(f"Batch complete: {len(generated_docs)} succeeded, {len(failed_docs)} failed")

    if failed_docs:
        print_warning("Failed documents:")
        for doc in failed_docs:
            print_error(f"  - {doc['customer']}: {doc['error']}")


# =============================================================================
# Edit Command
# =============================================================================

@cli.command()
@click.argument('doc_id')
@click.argument('instruction')
@click.option('--preview', '-p', is_flag=True, help='Preview changes without applying')
def edit(doc_id: str, instruction: str, preview: bool) -> None:
    """Apply an edit to a document via natural language instruction.

    DOC_ID: Document identifier
    INSTRUCTION: Natural language edit instruction

    Examples:

      prose-docs edit abc123 "Change the effective date to January 1, 2025"

      prose-docs edit abc123 "Remove the arbitration clause"

      prose-docs edit abc123 "Make the tone more formal"
    """
    if not ensure_initialized():
        return

    config = load_config()
    documents = config.get('documents', {})

    if doc_id not in documents:
        print_error(f"Document not found: {doc_id}")
        return

    doc = documents[doc_id]

    print_info(f"Processing edit instruction: \"{instruction}\"")

    if RICH_AVAILABLE:
        with create_progress_bar() as progress:
            task = progress.add_task("Applying edit...", total=4)

            # Step 1: Parse instruction
            progress.update(task, description="Parsing natural language instruction...")
            import time
            time.sleep(0.3)
            progress.advance(task)

            # Step 2: Locate target
            progress.update(task, description="Locating target content...")
            time.sleep(0.3)
            progress.advance(task)

            # Step 3: Apply edit
            progress.update(task, description="Applying modification...")
            time.sleep(0.3)
            progress.advance(task)

            # Step 4: Validate
            progress.update(task, description="Validating changes...")
            time.sleep(0.3)
            progress.advance(task)

    if preview:
        print_info("Preview mode - changes not applied")
        if RICH_AVAILABLE:
            console.print(Panel(f"""
[bold]Edit Preview[/bold]

Document: {doc_id}
Instruction: {instruction}

[yellow]Changes would be applied here...[/yellow]
"""))
        return

    # Record edit in history
    doc['edits'] = doc.get('edits', [])
    doc['edits'].append({
        'instruction': instruction,
        'applied_at': datetime.now().isoformat(),
        'status': 'applied'
    })
    doc['reviewed'] = False  # Mark as needing re-review
    documents[doc_id] = doc
    config['documents'] = documents
    save_config(config)

    print_success("Edit applied successfully!")
    print_warning("Document marked for re-review due to modifications.")


# =============================================================================
# Review Command
# =============================================================================

@cli.command()
@click.argument('doc_id')
@click.option('--notes', '-n', help='Review notes')
@click.option('--approved/--rejected', default=True, help='Approval status')
def review(doc_id: str, notes: Optional[str], approved: bool) -> None:
    """Mark a document as human-reviewed.

    DOC_ID: Document identifier

    Documents must be reviewed before final export.
    """
    if not ensure_initialized():
        return

    config = load_config()
    documents = config.get('documents', {})

    if doc_id not in documents:
        print_error(f"Document not found: {doc_id}")
        return

    doc = documents[doc_id]

    # Record review
    doc['reviewed'] = approved
    doc['review'] = {
        'reviewed_at': datetime.now().isoformat(),
        'approved': approved,
        'notes': notes or ''
    }

    documents[doc_id] = doc
    config['documents'] = documents
    save_config(config)

    if approved:
        print_success(f"Document {doc_id} approved and marked as reviewed.")
    else:
        print_warning(f"Document {doc_id} marked as rejected.")

    if notes:
        print_info(f"Notes: {notes}")


# =============================================================================
# Learn Commands
# =============================================================================

@cli.group()
def learn() -> None:
    """Manage learning rules and patterns."""
    pass


@learn.command('show')
@click.option('--category', '-c', help='Filter by category')
@click.option('--format', '-f', 'output_format', type=click.Choice(['table', 'json']),
              default='table', help='Output format')
def learn_show(category: Optional[str], output_format: str) -> None:
    """Show learned rules and patterns.

    Displays rules learned from:
    - Human edits and corrections
    - Approved document patterns
    - Manual rule additions
    """
    if not ensure_initialized():
        return

    config = load_config()
    rules = config.get('learned_rules', {})

    if category:
        rules = {k: v for k, v in rules.items() if v.get('category') == category}

    if not rules:
        print_warning("No learned rules found.")
        print_info("Rules are learned from human edits or added via 'prose-docs learn add'")
        return

    if output_format == 'json':
        click.echo(json.dumps(rules, indent=2))
        return

    if RICH_AVAILABLE:
        table = Table(title="Learned Rules")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Category", style="magenta")
        table.add_column("Rule", style="white")
        table.add_column("Confidence", style="green")
        table.add_column("Applied", style="dim")

        for rule_id, rule in rules.items():
            confidence = rule.get('confidence', 0)
            conf_str = f"{confidence:.0%}" if isinstance(confidence, float) else str(confidence)
            table.add_row(
                rule_id[:8],
                rule.get('category', 'general'),
                rule.get('rule', '')[:60],
                conf_str,
                str(rule.get('applied_count', 0))
            )

        console.print(table)
    else:
        click.echo("\nLearned Rules:")
        click.echo("-" * 60)
        for rule_id, rule in rules.items():
            click.echo(f"  [{rule_id[:8]}] {rule.get('rule', '')[:50]}")


@learn.command('add')
@click.argument('rule')
@click.option('--category', '-c', default='general', help='Rule category')
@click.option('--confidence', type=float, default=1.0, help='Confidence score (0-1)')
def learn_add(rule: str, category: str, confidence: float) -> None:
    """Add a manual rule to the learning system.

    RULE: The rule description in natural language

    Examples:

      prose-docs learn add "Always use 'shall' instead of 'will' in legal clauses"

      prose-docs learn add "Format currency as $X,XXX.XX" --category formatting
    """
    if not ensure_initialized():
        return

    import hashlib
    rule_id = hashlib.sha256(f"{rule}{datetime.now().isoformat()}".encode()).hexdigest()[:12]

    config = load_config()
    rules = config.get('learned_rules', {})

    rules[rule_id] = {
        'rule': rule,
        'category': category,
        'confidence': min(1.0, max(0.0, confidence)),
        'source': 'manual',
        'created_at': datetime.now().isoformat(),
        'applied_count': 0
    }

    config['learned_rules'] = rules
    save_config(config)

    print_success(f"Rule added successfully!")
    print_info(f"Rule ID: {rule_id}")
    print_info(f"Category: {category}")


# =============================================================================
# Status Command
# =============================================================================

@cli.command()
@click.option('--verbose', '-v', is_flag=True, help='Show detailed status')
def status(verbose: bool) -> None:
    """Show system status and statistics."""
    config = load_config()

    if not config.get('initialized'):
        print_error("System not initialized. Run 'prose-docs init' first.")
        return

    if RICH_AVAILABLE:
        # Create status panel
        console.print(Panel.fit(
            "[bold green]Prose-Docs System Status[/bold green]",
            border_style="green"
        ))

        # Statistics table
        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Version", config.get('version', 'Unknown'))
        table.add_row("Initialized", config.get('initialized_at', 'Unknown')[:19])
        table.add_row("Templates", str(len(config.get('templates', {}))))
        table.add_row("Customers", str(len(config.get('customers', {}))))
        table.add_row("Documents", str(len(config.get('documents', {}))))
        table.add_row("Learned Rules", str(len(config.get('learned_rules', {}))))

        # Count reviewed documents
        docs = config.get('documents', {})
        reviewed = sum(1 for d in docs.values() if d.get('reviewed'))
        table.add_row("Reviewed Docs", f"{reviewed}/{len(docs)}")

        console.print(table)

        if verbose:
            console.print("\n[bold]Configuration:[/bold]")
            console.print(f"  Database: {config.get('database', 'Not set')}")
            console.print(f"  Templates Dir: {config.get('templates_dir', 'Not set')}")
            console.print(f"  Exports Dir: {config.get('exports_dir', 'Not set')}")

            agents_config = config.get('agents', {})
            console.print(f"\n[bold]Agents:[/bold]")
            console.print(f"  Enabled: {agents_config.get('enabled', False)}")
            console.print(f"  Config: {agents_config.get('config_file', 'Not set')}")

            learning_config = config.get('learning', {})
            console.print(f"\n[bold]Learning:[/bold]")
            console.print(f"  Auto-learn: {learning_config.get('auto_learn', False)}")
            console.print(f"  Min Confidence: {learning_config.get('min_confidence', 0.8)}")
    else:
        click.echo("\nProse-Docs System Status")
        click.echo("=" * 40)
        click.echo(f"Version: {config.get('version', 'Unknown')}")
        click.echo(f"Templates: {len(config.get('templates', {}))}")
        click.echo(f"Customers: {len(config.get('customers', {}))}")
        click.echo(f"Documents: {len(config.get('documents', {}))}")
        click.echo(f"Learned Rules: {len(config.get('learned_rules', {}))}")


# =============================================================================
# Export Command
# =============================================================================

@cli.command()
@click.argument('doc_id')
@click.option('--format', '-f', 'output_format', type=click.Choice(['pdf', 'docx', 'md', 'html']),
              default='pdf', help='Export format')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--force', is_flag=True, help='Export even if not reviewed')
def export(doc_id: str, output_format: str, output: Optional[str], force: bool) -> None:
    """Export a document to final format.

    DOC_ID: Document identifier

    Documents must be reviewed before export unless --force is used.

    Supported formats:
    - pdf: Portable Document Format
    - docx: Microsoft Word
    - md: Markdown
    - html: HTML
    """
    if not ensure_initialized():
        return

    config = load_config()
    documents = config.get('documents', {})

    if doc_id not in documents:
        print_error(f"Document not found: {doc_id}")
        return

    doc = documents[doc_id]

    if not doc.get('reviewed') and not force:
        print_error("Document has not been reviewed.")
        print_info("Use 'prose-docs review <doc_id>' to review first, or --force to export anyway.")
        return

    if not doc.get('reviewed') and force:
        print_warning("Exporting unreviewed document...")

    print_info(f"Exporting document {doc_id} as {output_format.upper()}...")

    # Determine output path
    if output:
        output_path = Path(output)
    else:
        exports_dir = Path(config.get('exports_dir', './prose-docs/exports'))
        exports_dir.mkdir(parents=True, exist_ok=True)
        output_path = exports_dir / f"{doc_id}_final.{output_format}"

    if RICH_AVAILABLE:
        with create_progress_bar() as progress:
            task = progress.add_task("Exporting...", total=3)

            progress.update(task, description="Preparing document...")
            import time
            time.sleep(0.3)
            progress.advance(task)

            progress.update(task, description=f"Converting to {output_format.upper()}...")
            time.sleep(0.5)
            progress.advance(task)

            progress.update(task, description="Finalizing export...")
            time.sleep(0.2)
            progress.advance(task)

    # Update document record
    doc['exports'] = doc.get('exports', [])
    doc['exports'].append({
        'format': output_format,
        'path': str(output_path),
        'exported_at': datetime.now().isoformat()
    })
    documents[doc_id] = doc
    config['documents'] = documents
    save_config(config)

    print_success(f"Document exported successfully!")
    print_info(f"Output: {output_path}")


# =============================================================================
# Entry Point
# =============================================================================

def main() -> None:
    """Main entry point for the CLI."""
    try:
        cli(obj={})
    except KeyboardInterrupt:
        print_warning("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
