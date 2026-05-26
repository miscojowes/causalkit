#!/usr/bin/env python3
"""
Parse BIF (Bayesian Interchange Format) files to:
1. Extract the DAG structure (adjacency matrix)
2. Generate synthetic data by forward sampling

Usage: python3 parse_and_sample.py <bif_file> [num_samples] [output_prefix]
"""

import re
import sys
import os
import csv
import random


def parse_bif(filepath):
    """Parse a BIF file and return variables, domains, edges, and CPTs."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Remove comments
    content = re.sub(r'//.*', '', content)
    
    variables = []
    var_domains = {}  # variable -> list of values
    edges = []  # list of (parent, child)
    
    # CPT data: child_var -> {'parents': [p1, p2, ...], 'entries': {parent_combo: [probs]}}
    cpts = {}
    
    # Extract variable declarations
    var_pattern = r'variable\s+(\w[\w.()]*)\s*\{[^}]*type\s+discrete\s*\[\s*(\d+)\s*\]\s*\{([^}]+)\}'
    for match in re.finditer(var_pattern, content):
        name = match.group(1)
        size = int(match.group(2))
        values = [v.strip() for v in match.group(3).split(',')]
        variables.append(name)
        var_domains[name] = values
    
    # Sort variables topologically by their order of appearance (respects BIF convention)
    
    # Extract probability statements
    prob_pattern = r'probability\s*\(\s*(\w[\w.()]*)\s*(?:\|\s*([^)]+))?\s*\)\s*\{'
    
    for match in re.finditer(prob_pattern, content):
        child = match.group(1)
        if child not in variables:
            continue
        
        parents_str = match.group(2)
        if parents_str:
            parents = [p.strip() for p in parents_str.split(',')]
            for p in parents:
                if (p, child) not in edges:
                    edges.append((p, child))
        else:
            parents = []
        
        # Now extract the probability table values
        # Find the matching closing brace
        start = match.end()
        brace_depth = 1
        pos = start
        while pos < len(content) and brace_depth > 0:
            if content[pos] == '{':
                brace_depth += 1
            elif content[pos] == '}':
                brace_depth -= 1
            pos += 1
        
        table_content = content[start:pos-1]
        
        # Parse the probability table
        cpt_entry = {'parents': parents, 'raw': table_content}
        
        if not parents:
            # Try simple "table" format
            table_match = re.search(r'table\s+(.+?);', table_content)
            if table_match:
                probs_str = table_match.group(1)
                probs = [float(x.strip()) for x in probs_str.replace(';', '').split(',')]
                cpt_entry['type'] = 'table'
                cpt_entry['probs'] = probs
            else:
                cpt_entry['type'] = 'entries'
                cpt_entry['entries'] = parse_table_entries(table_content)
        else:
            cpt_entry['type'] = 'entries'
            cpt_entry['entries'] = parse_table_entries(table_content)
        
        cpts[child] = cpt_entry
    
    return variables, var_domains, edges, cpts


def parse_table_entries(table_content):
    """
    Parse a BIF probability table into a dictionary mapping
    parent_value_combinations -> list of child probabilities.
    """
    entries = {}
    
    # Pattern: (val1, val2, ...) prob1, prob2, ...;
    # Some tables don't have semicolons after each line
    entry_pattern = r'\(([^)]*)\)\s*([\d.eE+,\-\s\nd]+?)(?:\s*;|$)'
    
    for entry_match in re.finditer(entry_pattern, table_content):
        combo_str = entry_match.group(1).strip()
        probs_str = entry_match.group(2).strip().rstrip(';')
        
        combo = tuple(v.strip() for v in combo_str.split(','))
        # Parse probabilities
        probs = [float(x.strip()) for x in probs_str.replace(',', ' ').split()]
        entries[combo] = probs
    
    # Also check for "table" keyword in combined content
    table_match = re.search(r'table\s+(.+?);', table_content)
    if table_match and not entries:
        probs_str = table_match.group(1)
        probs = [float(x.strip()) for x in probs_str.split(',')]
        entries['__table__'] = probs
    
    return entries


def build_topological_order(variables, edges):
    """Build topological ordering of variables using Kahn's algorithm."""
    graph = {}
    in_degree = {}
    
    for v in variables:
        graph[v] = []
        in_degree[v] = 0
    
    for parent, child in edges:
        if parent in graph and child in graph:
            graph[parent].append(child)
            in_degree[child] = in_degree.get(child, 0) + 1
    
    queue = [v for v in variables if in_degree.get(v, 0) == 0]
    order = []
    
    while queue:
        v = queue.pop(0)
        order.append(v)
        for neighbor in graph[v]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    # If not all variables are in order (cycle), add remaining
    for v in variables:
        if v not in order:
            order.append(v)
    
    return order


def build_adjacency_matrix(variables, edges):
    """Build adjacency matrix (n_vars x n_vars) from edges."""
    n = len(variables)
    var_to_idx = {name: i for i, name in enumerate(variables)}
    adj = [[0] * n for _ in range(n)]
    
    for parent, child in edges:
        if parent in var_to_idx and child in var_to_idx:
            adj[var_to_idx[parent]][var_to_idx[child]] = 1
    
    return adj, var_to_idx


def save_adjacency_csv(adj, variables, filepath):
    """Save adjacency matrix as CSV with variable names."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([''] + variables)
        for i, var in enumerate(variables):
            writer.writerow([var] + adj[i])


def sample_from_bn(variables, var_domains, edges, cpts, num_samples, seed=42):
    """Generate samples by forward sampling from the Bayesian network."""
    random.seed(seed)
    
    # Build topological order
    order = build_topological_order(variables, edges)
    
    # Build parent lookup
    parents_of = {v: [] for v in variables}
    for parent, child in edges:
        parents_of[child].append(parent)
    
    # Validate CPTs
    for var in variables:
        if var in cpts:
            cpt = cpts[var]
            if cpt['type'] == 'entries':
                parents = cpt['parents']
                entries = cpt['entries']
                if entries and '__table__' in entries:
                    cpt['type'] = 'table'
                    cpt['probs'] = entries['__table__']
    
    # Generate samples
    samples = []
    for _ in range(num_samples):
        sample = {}
        for var in order:
            if var not in cpts:
                # No CPT found — pick a random value
                sample[var] = random.choice(var_domains[var])
                continue
            
            cpt = cpts[var]
            
            if cpt['type'] == 'table':
                # No parents — sample from flat table
                probs = cpt['probs']
                values = var_domains[var]
                sample[var] = random.choices(values, weights=probs, k=1)[0]
            elif cpt['type'] == 'entries':
                parents = cpt['parents']
                entries = cpt['entries']
                if not parents:
                    # Unconditional but with entries format (shouldn't happen, but handle it)
                    sample[var] = random.choice(var_domains[var])
                else:
                    # Has parents — lookup by parent values
                    parent_values = tuple(sample[p] for p in parents)
                    if parent_values in entries:
                        probs = entries[parent_values]
                        values = var_domains[var]
                        sample[var] = random.choices(values, weights=probs, k=1)[0]
                    else:
                        # Try partial matching - some BIF files have entries per parent value
                        # Fallback: pick uniformly
                        sample[var] = random.choice(var_domains[var])
            else:
                sample[var] = random.choice(var_domains[var])
        
        samples.append(sample)
    
    return order, samples


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 parse_and_sample.py <bif_file> [num_samples] [output_prefix]")
        sys.exit(1)
    
    bif_file = sys.argv[1]
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
    output_prefix = sys.argv[3] if len(sys.argv) > 3 else os.path.splitext(os.path.basename(bif_file))[0]
    
    # Parse BIF file
    print(f"Parsing {bif_file}...")
    variables, var_domains, edges, cpts = parse_bif(bif_file)
    
    print(f"  Variables ({len(variables)}): {variables}")
    print(f"  Edges ({len(edges)}):")
    for p, c in edges:
        print(f"    {p} -> {c}")
    
    # Build adjacency matrix
    adj, var_to_idx = build_adjacency_matrix(variables, edges)
    edge_count = sum(sum(row) for row in adj)
    print(f"\n  Total directed edges: {edge_count}")
    
    # Save adjacency matrix
    adj_file = f"{output_prefix}_dag.csv"
    save_adjacency_csv(adj, variables, adj_file)
    print(f"  Saved adjacency matrix: {adj_file}")
    
    # Generate samples
    print(f"\nGenerating {num_samples} samples via forward sampling...")
    order, samples = sample_from_bn(variables, var_domains, edges, cpts, num_samples)
    
    # Save samples as CSV
    data_file = f"{output_prefix}_data.csv"
    with open(data_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(order)
        for sample in samples:
            writer.writerow([sample[var] for var in order])
    
    print(f"  Saved data: {data_file}")
    print(f"\nDone! Variables: {len(variables)}, Edges: {edge_count}, Samples: {num_samples}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY FOR {output_prefix}")
    print(f"{'='*60}")
    print(f"Variables: {len(variables)}")
    print(f"  Names: {', '.join(variables)}")
    print(f"Edges: {edge_count}")
    print(f"Samples generated: {num_samples}")
    print(f"Files created:")
    print(f"  {adj_file}")
    print(f"  {data_file}")


if __name__ == '__main__':
    main()
