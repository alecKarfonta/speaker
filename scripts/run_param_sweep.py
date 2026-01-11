#!/usr/bin/env python3
"""
TTS Parameter Sweep - Automated testing of different TTS configurations.

This script runs the validation suite across multiple parameter combinations,
restarting the TTS container for each configuration and collecting results.

Usage:
    ./scripts/run_param_sweep.py
    
    # Quick sweep (fewer params, fast validation)
    ./scripts/run_param_sweep.py --quick
    
    # Custom sweep file
    ./scripts/run_param_sweep.py --config sweep_config.json

Output:
    - tests/output/sweep_results.json: Full results data
    - tests/output/sweep_report.txt: Human-readable summary
    - tests/output/sweep_TIMESTAMP/: Per-run validation reports
"""

import os
import sys
import json
import time
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import argparse

# Configuration
DOCKER_COMPOSE_PATH = Path(__file__).parent.parent / "docker-compose.yml"
OUTPUT_DIR = Path(__file__).parent.parent / "tests" / "output"
TTS_API_URL = os.environ.get("TTS_API_URL", "http://localhost:8012")
STT_API_URL = os.environ.get("STT_API_URL", "http://192.168.1.77:8603/v1/audio/transcriptions")
STT_API_KEY = os.environ.get("STT_API_KEY", "stt-api-key")

# Health check settings
MAX_STARTUP_WAIT = 600  # 10 minutes max wait for startup
HEALTH_CHECK_INTERVAL = 15  # Check every 15 seconds
HEALTH_CHECK_RETRIES = 5  # Require 5 consecutive healthy checks


def log(msg: str):
    """Print timestamped log message."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")


def run_command(cmd: List[str], cwd: Optional[Path] = None, capture: bool = False) -> subprocess.CompletedProcess:
    """Run a shell command."""
    if capture:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    else:
        return subprocess.run(cmd, cwd=cwd)


def wait_for_healthy(timeout: int = MAX_STARTUP_WAIT) -> bool:
    """
    Wait for TTS API to become healthy.
    Returns True if healthy, False if timeout.
    """
    import requests
    
    start_time = time.time()
    consecutive_healthy = 0
    
    log(f"Waiting for TTS API to become healthy (max {timeout}s)...")
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{TTS_API_URL}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    consecutive_healthy += 1
                    elapsed = int(time.time() - start_time)
                    log(f"  Health check passed ({consecutive_healthy}/{HEALTH_CHECK_RETRIES}) - {elapsed}s elapsed")
                    
                    if consecutive_healthy >= HEALTH_CHECK_RETRIES:
                        log(f"✅ TTS API healthy after {elapsed}s")
                        return True
                else:
                    consecutive_healthy = 0
            else:
                consecutive_healthy = 0
        except Exception:
            consecutive_healthy = 0
            elapsed = int(time.time() - start_time)
            if elapsed % 60 == 0:  # Log every minute
                log(f"  Still waiting... ({elapsed}s)")
        
        time.sleep(HEALTH_CHECK_INTERVAL)
    
    log(f"❌ Timeout waiting for TTS API after {timeout}s")
    return False


def update_docker_compose_env(params: Dict[str, str]) -> None:
    """
    Update docker-compose.yml with new environment variables.
    """
    with open(DOCKER_COMPOSE_PATH, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    new_lines = []
    in_tts_env = False
    env_indent = None
    
    for line in lines:
        # Detect when we're in the tts-api environment section
        if 'tts-api:' in line:
            in_tts_env = False
        if 'environment:' in line and in_tts_env is False:
            # Check if this is the tts-api environment by looking back
            for prev_line in reversed(new_lines[-10:]):
                if 'tts-api:' in prev_line:
                    in_tts_env = True
                    break
                if 'services:' in prev_line or any(svc in prev_line for svc in ['frontend:', 'prometheus:', 'grafana:']):
                    break
        
        if in_tts_env and 'volumes:' in line:
            in_tts_env = False
        
        # Check if this line is an env var we want to update
        updated = False
        if in_tts_env and env_indent is None and line.strip().startswith('- '):
            env_indent = len(line) - len(line.lstrip())
        
        if in_tts_env and line.strip().startswith('- '):
            for param_name, param_value in params.items():
                if f'{param_name}=' in line or f'{param_name}:' in line:
                    # Replace this line
                    new_lines.append(f"{' ' * env_indent}- {param_name}={param_value}")
                    updated = True
                    break
        
        if not updated:
            new_lines.append(line)
    
    with open(DOCKER_COMPOSE_PATH, 'w') as f:
        f.write('\n'.join(new_lines))


def read_docker_compose_env() -> Dict[str, str]:
    """
    Read current GLM_TTS environment variables from docker-compose.yml.
    Returns dict of all GLM_TTS_* settings.
    """
    with open(DOCKER_COMPOSE_PATH, 'r') as f:
        content = f.read()
    
    config = {}
    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('- GLM_TTS_'):
            # Parse "- GLM_TTS_XYZ=value" or "- GLM_TTS_XYZ=value # comment"
            if '=' in line:
                key_val = line[2:].split('#')[0].strip()  # Remove "- " prefix and comments
                if '=' in key_val:
                    key, val = key_val.split('=', 1)
                    config[key] = val
    
    return config

def restart_tts_container() -> bool:
    """
    Restart the TTS container and wait for it to become healthy.
    """
    log("Stopping TTS container...")
    run_command(["docker", "compose", "stop", "tts-api"], cwd=DOCKER_COMPOSE_PATH.parent)
    
    log("Starting TTS container with new configuration...")
    run_command(["docker", "compose", "up", "-d", "tts-api"], cwd=DOCKER_COMPOSE_PATH.parent)
    
    # Wait for container to become healthy
    return wait_for_healthy()


def run_validation(run_name: str) -> Dict[str, Any]:
    """
    Run the validation script and collect results.
    Returns a dict with performance and quality metrics.
    """
    log(f"Running validation for: {run_name}")
    
    # Create per-run output directory
    run_dir = OUTPUT_DIR / f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Run validation script
    env = os.environ.copy()
    env["TTS_API_URL"] = TTS_API_URL
    env["STT_API_URL"] = STT_API_URL
    env["STT_API_KEY"] = STT_API_KEY
    
    # Use venv Python to ensure dependencies are available
    venv_python = DOCKER_COMPOSE_PATH.parent / ".venv" / "bin" / "python3"
    python_exe = str(venv_python) if venv_python.exists() else sys.executable
    
    result = subprocess.run(
        [python_exe, "scripts/run_tts_validation.py"],
        cwd=DOCKER_COMPOSE_PATH.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800  # 30 minute timeout
    )
    
    # Save raw output
    with open(run_dir / "validation_output.txt", "w") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n\nSTDERR:\n")
            f.write(result.stderr)
    
    # Copy validation report
    report_src = OUTPUT_DIR / "validation_report.txt"
    if report_src.exists():
        shutil.copy(report_src, run_dir / "validation_report.txt")
    
    # Parse results from output
    metrics = parse_validation_output(result.stdout)
    metrics["run_name"] = run_name
    metrics["run_dir"] = str(run_dir)
    metrics["exit_code"] = result.returncode
    
    return metrics


def parse_validation_output(output: str) -> Dict[str, Any]:
    """Parse metrics from validation script output."""
    metrics = {
        "speed": 0.0,
        "total_audio": 0.0,
        "compute_time": 0.0,
        "accuracy": 0.0,
        "tests_passed": 0,
        "tests_total": 0,
        "tests_failed": 0,
    }
    
    for line in output.split('\n'):
        if 'Overall Speed:' in line:
            try:
                # Extract "4.62x" -> 4.62
                parts = line.split('Speed:')[1].strip().split('x')[0]
                metrics["speed"] = float(parts)
            except:
                pass
        
        if 'Total Audio Generated:' in line or 'Total Audio:' in line:
            try:
                parts = line.split(':')[1].strip().split('s')[0]
                metrics["total_audio"] = float(parts)
            except:
                pass
        
        if 'Total Compute Time:' in line or 'compute' in line.lower():
            try:
                if 'in' in line:
                    # "26.79s in 5.79s compute"
                    parts = line.split('in')[1].strip().split('s')[0]
                else:
                    parts = line.split(':')[1].strip().split('s')[0]
                metrics["compute_time"] = float(parts)
            except:
                pass
        
        if 'Average Accuracy:' in line or 'Average Similarity:' in line:
            try:
                parts = line.split(':')[1].strip().rstrip('%')
                metrics["accuracy"] = float(parts)
            except:
                pass
        
        if 'Tests Passed:' in line:
            try:
                parts = line.split(':')[1].strip()
                passed, total = parts.split('/')
                metrics["tests_passed"] = int(passed)
                metrics["tests_total"] = int(total)
            except:
                pass
        
        if 'Tests Failed:' in line:
            try:
                metrics["tests_failed"] = int(line.split(':')[1].strip())
            except:
                pass
    
    return metrics


def generate_report(results: List[Dict[str, Any]], output_path: Path) -> None:
    """Generate human-readable sweep report."""
    lines = [
        "=" * 80,
        "  TTS PARAMETER SWEEP RESULTS",
        "=" * 80,
        f"  Timestamp: {datetime.now().isoformat()}",
        f"  Total configurations tested: {len(results)}",
        "",
        "-" * 80,
        "  RESULTS BY SPEED (fastest first)",
        "-" * 80,
        "",
    ]
    
    # Sort by speed
    sorted_results = sorted(results, key=lambda x: x.get("speed", 0), reverse=True)
    
    lines.append(f"  {'Config':<30} {'Speed':>8} {'Accuracy':>10} {'Passed':>8} {'Failed':>8}")
    lines.append(f"  {'-'*30} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")
    
    for r in sorted_results:
        name = r.get("run_name", "unknown")[:30]
        speed = f"{r.get('speed', 0):.2f}x"
        accuracy = f"{r.get('accuracy', 0):.1f}%"
        passed = str(r.get("tests_passed", 0))
        failed = str(r.get("tests_failed", 0))
        lines.append(f"  {name:<30} {speed:>8} {accuracy:>10} {passed:>8} {failed:>8}")
    
    lines.extend([
        "",
        "-" * 80,
        "  BEST CONFIGURATIONS",
        "-" * 80,
        "",
    ])
    
    if sorted_results:
        best_speed = sorted_results[0]
        lines.append(f"  🏆 Fastest: {best_speed['run_name']} ({best_speed['speed']:.2f}x)")
        
        best_quality = max(results, key=lambda x: x.get("accuracy", 0))
        lines.append(f"  🎯 Best Quality: {best_quality['run_name']} ({best_quality['accuracy']:.1f}% accuracy)")
        
        # Best balanced (speed * accuracy)
        for r in results:
            r["score"] = r.get("speed", 0) * r.get("accuracy", 0) / 100
        best_balanced = max(results, key=lambda x: x.get("score", 0))
        lines.append(f"  ⚖️ Best Balanced: {best_balanced['run_name']} (speed={best_balanced['speed']:.2f}x, acc={best_balanced['accuracy']:.1f}%)")
    
    lines.extend(["", "=" * 80])
    
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    
    # Also print to console
    print("\n".join(lines))
    
    # Generate charts
    generate_charts(results, output_path.parent)


def generate_charts(results: List[Dict[str, Any]], output_dir: Path) -> None:
    """Generate professional performance comparison charts."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        log("⚠️ matplotlib not available, skipping chart generation")
        return
    
    # Filter out failed runs
    valid_results = [r for r in results if r.get("speed", 0) > 0 and r.get("accuracy", 0) > 0]
    
    if len(valid_results) < 2:
        log("⚠️ Not enough valid results for charts")
        return
    
    log("Generating performance charts...")
    
    # Use a professional dark theme
    plt.style.use('dark_background')
    
    # Color palette
    colors = {
        'primary': '#00D4AA',      # Teal
        'secondary': '#FF6B6B',    # Coral
        'accent': '#4ECDC4',       # Light teal
        'highlight': '#FFE66D',    # Yellow
        'muted': '#95A5A6',        # Gray
        'bg': '#1a1a2e',           # Dark blue-gray
    }
    
    # Extract data
    names = [r.get("run_name", "?")[:20] for r in valid_results]
    speeds = [r.get("speed", 0) for r in valid_results]
    accuracies = [r.get("accuracy", 0) for r in valid_results]
    scores = [s * a / 100 for s, a in zip(speeds, accuracies)]
    
    # Create figure with 2x2 subplots
    fig = plt.figure(figsize=(16, 12), facecolor='#0d0d1a')
    fig.suptitle('TTS Parameter Sweep Results', fontsize=20, fontweight='bold', 
                 color='white', y=0.98)
    
    # 1. Speed vs Accuracy Scatter Plot with Pareto Frontier
    ax1 = fig.add_subplot(2, 2, 1, facecolor='#1a1a2e')
    
    # Calculate Pareto frontier
    pareto_points = []
    for i, (s, a) in enumerate(zip(speeds, accuracies)):
        is_pareto = True
        for j, (s2, a2) in enumerate(zip(speeds, accuracies)):
            if i != j and s2 >= s and a2 >= a and (s2 > s or a2 > a):
                is_pareto = False
                break
        if is_pareto:
            pareto_points.append(i)
    
    # Plot all points
    scatter = ax1.scatter(speeds, accuracies, c=scores, cmap='viridis', 
                          s=100, alpha=0.7, edgecolors='white', linewidths=0.5)
    
    # Highlight Pareto optimal points
    pareto_speeds = [speeds[i] for i in pareto_points]
    pareto_accs = [accuracies[i] for i in pareto_points]
    ax1.scatter(pareto_speeds, pareto_accs, c=colors['highlight'], s=200, 
                marker='*', edgecolors='white', linewidths=1, zorder=5,
                label='Pareto Optimal')
    
    # Draw Pareto frontier line
    if len(pareto_points) > 1:
        sorted_pareto = sorted(zip(pareto_speeds, pareto_accs), key=lambda x: x[0])
        px, py = zip(*sorted_pareto)
        ax1.plot(px, py, '--', color=colors['highlight'], alpha=0.5, linewidth=2)
    
    ax1.set_xlabel('Speed (x real-time)', fontsize=12, color='white')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, color='white')
    ax1.set_title('Speed vs Quality Trade-off', fontsize=14, fontweight='bold', color='white', pad=15)
    ax1.grid(True, alpha=0.2, color='white')
    ax1.legend(loc='lower right', facecolor='#1a1a2e', edgecolor='white')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1, label='Combined Score')
    cbar.ax.yaxis.label.set_color('white')
    cbar.ax.tick_params(colors='white')
    
    # 2. Speed Bar Chart (Top 15)
    ax2 = fig.add_subplot(2, 2, 2, facecolor='#1a1a2e')
    
    # Sort by speed and take top 15
    sorted_by_speed = sorted(zip(names, speeds, accuracies), key=lambda x: x[1], reverse=True)[:15]
    bar_names, bar_speeds, bar_accs = zip(*sorted_by_speed) if sorted_by_speed else ([], [], [])
    
    y_pos = np.arange(len(bar_names))
    
    # Create gradient colors based on accuracy
    bar_colors = [plt.cm.RdYlGn(a / 100) for a in bar_accs]
    
    bars = ax2.barh(y_pos, bar_speeds, color=bar_colors, edgecolor='white', linewidth=0.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(bar_names, fontsize=9, color='white')
    ax2.invert_yaxis()
    ax2.set_xlabel('Speed (x real-time)', fontsize=12, color='white')
    ax2.set_title('Top 15 Configurations by Speed', fontsize=14, fontweight='bold', color='white', pad=15)
    ax2.grid(True, alpha=0.2, axis='x', color='white')
    
    # Add speed values on bars
    for i, (bar, speed, acc) in enumerate(zip(bars, bar_speeds, bar_accs)):
        ax2.text(speed + 0.1, i, f'{speed:.1f}x ({acc:.0f}%)', 
                va='center', fontsize=8, color='white')
    
    # 3. Parameter Impact Analysis (if params available)
    ax3 = fig.add_subplot(2, 2, 3, facecolor='#1a1a2e')
    
    # Analyze parameter impact on speed
    param_impacts = {}
    key_params = ['GLM_TTS_FLOW_STEPS', 'GLM_TTS_CFG_RATE', 'GLM_TTS_SAMPLING', 
                  'GLM_TTS_QUANTIZATION', 'GLM_TTS_COMPILE_FLOW']
    
    for param in key_params:
        param_values = {}
        for r in valid_results:
            params = r.get("params", {})
            if param in params:
                val = params[param]
                if val not in param_values:
                    param_values[val] = []
                param_values[val].append(r.get("speed", 0))
        
        if param_values:
            avg_speeds = {v: np.mean(speeds) for v, speeds in param_values.items()}
            param_impacts[param.replace("GLM_TTS_", "")] = avg_speeds
    
    if param_impacts:
        # Create grouped bar chart for parameter impacts
        param_names = list(param_impacts.keys())
        x = np.arange(len(param_names))
        width = 0.15
        
        # Get all unique values across parameters
        all_values = set()
        for impacts in param_impacts.values():
            all_values.update(impacts.keys())
        all_values = sorted(all_values)[:6]  # Limit to 6 values
        
        value_colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(all_values)))
        
        for i, val in enumerate(all_values):
            val_speeds = []
            for param in param_names:
                if val in param_impacts[param]:
                    val_speeds.append(param_impacts[param][val])
                else:
                    val_speeds.append(0)
            
            offset = (i - len(all_values)/2) * width
            bars = ax3.bar(x + offset, val_speeds, width, label=str(val)[:10], 
                          color=value_colors[i], edgecolor='white', linewidth=0.5)
        
        ax3.set_ylabel('Avg Speed (x real-time)', fontsize=12, color='white')
        ax3.set_xticks(x)
        ax3.set_xticklabels(param_names, fontsize=9, rotation=45, ha='right', color='white')
        ax3.set_title('Parameter Impact on Speed', fontsize=14, fontweight='bold', color='white', pad=15)
        ax3.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='white', 
                   fontsize=8, ncol=2, title='Value')
        ax3.grid(True, alpha=0.2, axis='y', color='white')
    else:
        ax3.text(0.5, 0.5, 'Parameter data not available', transform=ax3.transAxes,
                ha='center', va='center', fontsize=14, color='white')
    
    # 4. Score Distribution (Speed × Accuracy)
    ax4 = fig.add_subplot(2, 2, 4, facecolor='#1a1a2e')
    
    # Sort by combined score
    sorted_by_score = sorted(zip(names, scores, speeds, accuracies), key=lambda x: x[1], reverse=True)[:15]
    
    if sorted_by_score:
        score_names, score_vals, score_speeds, score_accs = zip(*sorted_by_score)
        y_pos = np.arange(len(score_names))
        
        # Color based on what the config optimizes for
        bar_colors_score = []
        for s, a in zip(score_speeds, score_accs):
            if s > np.median(speeds) and a > np.median(accuracies):
                bar_colors_score.append(colors['highlight'])  # Balanced
            elif s > np.median(speeds):
                bar_colors_score.append(colors['primary'])   # Speed-focused
            else:
                bar_colors_score.append(colors['secondary'])  # Quality-focused
        
        bars = ax4.barh(y_pos, score_vals, color=bar_colors_score, 
                       edgecolor='white', linewidth=0.5)
        
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(score_names, fontsize=9, color='white')
        ax4.invert_yaxis()
        ax4.set_xlabel('Combined Score (Speed × Accuracy)', fontsize=12, color='white')
        ax4.set_title('Top 15 by Overall Performance', fontsize=14, fontweight='bold', color='white', pad=15)
        ax4.grid(True, alpha=0.2, axis='x', color='white')
        
        # Legend
        legend_elements = [
            mpatches.Patch(facecolor=colors['highlight'], edgecolor='white', label='Balanced'),
            mpatches.Patch(facecolor=colors['primary'], edgecolor='white', label='Speed Focus'),
            mpatches.Patch(facecolor=colors['secondary'], edgecolor='white', label='Quality Focus'),
        ]
        ax4.legend(handles=legend_elements, loc='lower right', 
                  facecolor='#1a1a2e', edgecolor='white', fontsize=9)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    # Save charts
    chart_path = output_dir / "sweep_charts.png"
    plt.savefig(chart_path, dpi=150, facecolor='#0d0d1a', edgecolor='none',
                bbox_inches='tight')
    plt.close()
    
    log(f"📊 Charts saved to: {chart_path}")


def run_sweep(param_sets: List[Dict[str, Any]], quick: bool = False) -> List[Dict[str, Any]]:
    """
    Run the parameter sweep.
    
    Args:
        param_sets: List of parameter configurations to test
        quick: If True, run quick validation (fewer tests)
    
    Returns:
        List of results for each configuration
    """
    results = []
    total = len(param_sets)
    
    # Filter out comment-only entries
    param_sets = [p for p in param_sets if not (len(p) == 1 and "_comment" in p)]
    param_sets = [p for p in param_sets if "name" in p or any(k.startswith("GLM_TTS") for k in p)]
    total = len(param_sets)
    
    log(f"Starting parameter sweep with {total} configurations")
    log(f"Estimated time: {total * 12} - {total * 18} minutes")
    
    for i, param_set in enumerate(param_sets, 1):
        name = param_set.pop("name", f"config_{i}")
        
        # Remove any comment fields
        param_set = {k: v for k, v in param_set.items() if not k.startswith("_")}
        
        log("")
        log("=" * 60)
        log(f"Configuration {i}/{total}: {name}")
        log("Parameters being applied:")
        for k, v in sorted(param_set.items()):
            log(f"  {k}: {v}")
        log("=" * 60)
        
        try:
            # Update docker-compose with new params
            update_docker_compose_env(param_set)
            
            # Read back actual config from docker-compose for logging
            actual_config = read_docker_compose_env()
            
            # Restart container
            if not restart_tts_container():
                log(f"❌ Failed to start container for {name}, skipping...")
                results.append({
                    "run_name": name,
                    "params": actual_config,  # Store ACTUAL config
                    "error": "Container failed to start",
                    "speed": 0,
                    "accuracy": 0,
                })
                continue
            
            # Run validation
            metrics = run_validation(name)
            metrics["params"] = actual_config  # Store ACTUAL config, not just requested changes
            results.append(metrics)
            
            log(f"✅ {name}: Speed={metrics['speed']:.2f}x, Accuracy={metrics['accuracy']:.1f}%")
            
        except Exception as e:
            log(f"❌ Error running {name}: {e}")
            results.append({
                "run_name": name,
                "params": param_set,
                "error": str(e),
                "speed": 0,
                "accuracy": 0,
            })
        
        # Save intermediate results
        with open(OUTPUT_DIR / "sweep_results_partial.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
    
    return results


# Default parameter sweep configurations
# Each config tests a specific aspect of the pipeline
DEFAULT_PARAM_SETS = [
    # === BASELINE ===
    {
        "name": "baseline",
        "GLM_TTS_FLOW_STEPS": "3",
        "GLM_TTS_SAMPLING": "2",
        "GLM_TTS_CFG_RATE": "0.7",
        "GLM_TTS_QUANTIZATION": "4bit",
        "GLM_TTS_ATTENTION": "flash_attention_2",
        "GLM_TTS_VLLM_QUANTIZATION": "fp8",
        "GLM_TTS_FLOW_DTYPE": "fp16",
        "GLM_TTS_COMPILE_FLOW": "true",
        "GLM_TTS_COMPILE_VOCODER": "true",
    },
    
    # === FLOW STEPS (quality vs speed) ===
    {
        "name": "flow_steps_1",
        "GLM_TTS_FLOW_STEPS": "1",
        "GLM_TTS_SAMPLING": "2",
        "GLM_TTS_CFG_RATE": "0.7",
        "GLM_TTS_QUANTIZATION": "4bit",
        "GLM_TTS_ATTENTION": "flash_attention_2",
        "GLM_TTS_VLLM_QUANTIZATION": "fp8",
        "GLM_TTS_FLOW_DTYPE": "fp16",
        "GLM_TTS_COMPILE_FLOW": "true",
        "GLM_TTS_COMPILE_VOCODER": "true",
    },
    {
        "name": "flow_steps_5",
        "GLM_TTS_FLOW_STEPS": "5",
        "GLM_TTS_SAMPLING": "2",
        "GLM_TTS_CFG_RATE": "0.7",
        "GLM_TTS_QUANTIZATION": "4bit",
        "GLM_TTS_ATTENTION": "flash_attention_2",
        "GLM_TTS_VLLM_QUANTIZATION": "fp8",
        "GLM_TTS_FLOW_DTYPE": "fp16",
        "GLM_TTS_COMPILE_FLOW": "true",
        "GLM_TTS_COMPILE_VOCODER": "true",
    },
    {
        "name": "flow_steps_10",
        "GLM_TTS_FLOW_STEPS": "10",
        "GLM_TTS_SAMPLING": "2",
        "GLM_TTS_CFG_RATE": "0.7",
        "GLM_TTS_QUANTIZATION": "4bit",
        "GLM_TTS_ATTENTION": "flash_attention_2",
        "GLM_TTS_VLLM_QUANTIZATION": "fp8",
        "GLM_TTS_FLOW_DTYPE": "fp16",
        "GLM_TTS_COMPILE_FLOW": "true",
        "GLM_TTS_COMPILE_VOCODER": "true",
    },
    
    # === CFG RATE (0.0 = 2x faster flow, quality tradeoff) ===
    {
        "name": "cfg_disabled",
        "GLM_TTS_FLOW_STEPS": "3",
        "GLM_TTS_SAMPLING": "2",
        "GLM_TTS_CFG_RATE": "0.0",
        "GLM_TTS_QUANTIZATION": "4bit",
        "GLM_TTS_ATTENTION": "flash_attention_2",
        "GLM_TTS_VLLM_QUANTIZATION": "fp8",
        "GLM_TTS_FLOW_DTYPE": "fp16",
        "GLM_TTS_COMPILE_FLOW": "true",
        "GLM_TTS_COMPILE_VOCODER": "true",
    },
    {
        "name": "cfg_low_0.3",
        "GLM_TTS_FLOW_STEPS": "3",
        "GLM_TTS_SAMPLING": "2",
        "GLM_TTS_CFG_RATE": "0.3",
        "GLM_TTS_QUANTIZATION": "4bit",
        "GLM_TTS_ATTENTION": "flash_attention_2",
        "GLM_TTS_VLLM_QUANTIZATION": "fp8",
        "GLM_TTS_FLOW_DTYPE": "fp16",
        "GLM_TTS_COMPILE_FLOW": "true",
        "GLM_TTS_COMPILE_VOCODER": "true",
    },
    
    # === SAMPLING (top-k value) ===
    {
        "name": "sampling_1",
        "GLM_TTS_FLOW_STEPS": "3",
        "GLM_TTS_SAMPLING": "1",
        "GLM_TTS_CFG_RATE": "0.7",
        "GLM_TTS_QUANTIZATION": "4bit",
        "GLM_TTS_ATTENTION": "flash_attention_2",
        "GLM_TTS_VLLM_QUANTIZATION": "fp8",
        "GLM_TTS_FLOW_DTYPE": "fp16",
        "GLM_TTS_COMPILE_FLOW": "true",
        "GLM_TTS_COMPILE_VOCODER": "true",
    },
    {
        "name": "sampling_5",
        "GLM_TTS_FLOW_STEPS": "3",
        "GLM_TTS_SAMPLING": "5",
        "GLM_TTS_CFG_RATE": "0.7",
        "GLM_TTS_QUANTIZATION": "4bit",
        "GLM_TTS_ATTENTION": "flash_attention_2",
        "GLM_TTS_VLLM_QUANTIZATION": "fp8",
        "GLM_TTS_FLOW_DTYPE": "fp16",
        "GLM_TTS_COMPILE_FLOW": "true",
        "GLM_TTS_COMPILE_VOCODER": "true",
    },
    {
        "name": "sampling_15",
        "GLM_TTS_FLOW_STEPS": "3",
        "GLM_TTS_SAMPLING": "15",
        "GLM_TTS_CFG_RATE": "0.7",
        "GLM_TTS_QUANTIZATION": "4bit",
        "GLM_TTS_ATTENTION": "flash_attention_2",
        "GLM_TTS_VLLM_QUANTIZATION": "fp8",
        "GLM_TTS_FLOW_DTYPE": "fp16",
        "GLM_TTS_COMPILE_FLOW": "true",
        "GLM_TTS_COMPILE_VOCODER": "true",
    },
    
    # === QUANTIZATION OPTIONS ===
    {
        "name": "quant_8bit",
        "GLM_TTS_FLOW_STEPS": "3",
        "GLM_TTS_SAMPLING": "2",
        "GLM_TTS_CFG_RATE": "0.7",
        "GLM_TTS_QUANTIZATION": "8bit",
        "GLM_TTS_ATTENTION": "flash_attention_2",
        "GLM_TTS_VLLM_QUANTIZATION": "fp8",
        "GLM_TTS_FLOW_DTYPE": "fp16",
        "GLM_TTS_COMPILE_FLOW": "true",
        "GLM_TTS_COMPILE_VOCODER": "true",
    },
    {
        "name": "quant_none",
        "GLM_TTS_FLOW_STEPS": "3",
        "GLM_TTS_SAMPLING": "2",
        "GLM_TTS_CFG_RATE": "0.7",
        "GLM_TTS_QUANTIZATION": "none",
        "GLM_TTS_ATTENTION": "flash_attention_2",
        "GLM_TTS_VLLM_QUANTIZATION": "fp8",
        "GLM_TTS_FLOW_DTYPE": "fp16",
        "GLM_TTS_COMPILE_FLOW": "true",
        "GLM_TTS_COMPILE_VOCODER": "true",
    },
    {
        "name": "vllm_quant_none",
        "GLM_TTS_FLOW_STEPS": "3",
        "GLM_TTS_SAMPLING": "2",
        "GLM_TTS_CFG_RATE": "0.7",
        "GLM_TTS_QUANTIZATION": "4bit",
        "GLM_TTS_ATTENTION": "flash_attention_2",
        "GLM_TTS_VLLM_QUANTIZATION": "none",
        "GLM_TTS_FLOW_DTYPE": "fp16",
        "GLM_TTS_COMPILE_FLOW": "true",
        "GLM_TTS_COMPILE_VOCODER": "true",
    },
    
    # === ATTENTION OPTIONS ===
    {
        "name": "attention_sdpa",
        "GLM_TTS_FLOW_STEPS": "3",
        "GLM_TTS_SAMPLING": "2",
        "GLM_TTS_CFG_RATE": "0.7",
        "GLM_TTS_QUANTIZATION": "4bit",
        "GLM_TTS_ATTENTION": "sdpa",
        "GLM_TTS_VLLM_QUANTIZATION": "fp8",
        "GLM_TTS_FLOW_DTYPE": "fp16",
        "GLM_TTS_COMPILE_FLOW": "true",
        "GLM_TTS_COMPILE_VOCODER": "true",
    },
    {
        "name": "attention_eager",
        "GLM_TTS_FLOW_STEPS": "3",
        "GLM_TTS_SAMPLING": "2",
        "GLM_TTS_CFG_RATE": "0.7",
        "GLM_TTS_QUANTIZATION": "4bit",
        "GLM_TTS_ATTENTION": "eager",
        "GLM_TTS_VLLM_QUANTIZATION": "fp8",
        "GLM_TTS_FLOW_DTYPE": "fp16",
        "GLM_TTS_COMPILE_FLOW": "true",
        "GLM_TTS_COMPILE_VOCODER": "true",
    },
    
    # === FLOW DTYPE ===
    {
        "name": "flow_fp32",
        "GLM_TTS_FLOW_STEPS": "3",
        "GLM_TTS_SAMPLING": "2",
        "GLM_TTS_CFG_RATE": "0.7",
        "GLM_TTS_QUANTIZATION": "4bit",
        "GLM_TTS_ATTENTION": "flash_attention_2",
        "GLM_TTS_VLLM_QUANTIZATION": "fp8",
        "GLM_TTS_FLOW_DTYPE": "fp32",
        "GLM_TTS_COMPILE_FLOW": "true",
        "GLM_TTS_COMPILE_VOCODER": "true",
    },
    
    # === TORCH COMPILE OPTIONS ===
    {
        "name": "no_compile_flow",
        "GLM_TTS_FLOW_STEPS": "3",
        "GLM_TTS_SAMPLING": "2",
        "GLM_TTS_CFG_RATE": "0.7",
        "GLM_TTS_QUANTIZATION": "4bit",
        "GLM_TTS_ATTENTION": "flash_attention_2",
        "GLM_TTS_VLLM_QUANTIZATION": "fp8",
        "GLM_TTS_FLOW_DTYPE": "fp16",
        "GLM_TTS_COMPILE_FLOW": "false",
        "GLM_TTS_COMPILE_VOCODER": "true",
    },
    {
        "name": "no_compile_vocoder",
        "GLM_TTS_FLOW_STEPS": "3",
        "GLM_TTS_SAMPLING": "2",
        "GLM_TTS_CFG_RATE": "0.7",
        "GLM_TTS_QUANTIZATION": "4bit",
        "GLM_TTS_ATTENTION": "flash_attention_2",
        "GLM_TTS_VLLM_QUANTIZATION": "fp8",
        "GLM_TTS_FLOW_DTYPE": "fp16",
        "GLM_TTS_COMPILE_FLOW": "true",
        "GLM_TTS_COMPILE_VOCODER": "false",
    },
    {
        "name": "no_compile_both",
        "GLM_TTS_FLOW_STEPS": "3",
        "GLM_TTS_SAMPLING": "2",
        "GLM_TTS_CFG_RATE": "0.7",
        "GLM_TTS_QUANTIZATION": "4bit",
        "GLM_TTS_ATTENTION": "flash_attention_2",
        "GLM_TTS_VLLM_QUANTIZATION": "fp8",
        "GLM_TTS_FLOW_DTYPE": "fp16",
        "GLM_TTS_COMPILE_FLOW": "false",
        "GLM_TTS_COMPILE_VOCODER": "false",
    },
    
    # === COMBINED: Best speed attempt ===
    {
        "name": "speed_optimized",
        "GLM_TTS_FLOW_STEPS": "1",
        "GLM_TTS_SAMPLING": "1",
        "GLM_TTS_CFG_RATE": "0.0",
        "GLM_TTS_QUANTIZATION": "4bit",
        "GLM_TTS_ATTENTION": "flash_attention_2",
        "GLM_TTS_VLLM_QUANTIZATION": "fp8",
        "GLM_TTS_FLOW_DTYPE": "fp16",
        "GLM_TTS_COMPILE_FLOW": "true",
        "GLM_TTS_COMPILE_VOCODER": "true",
    },
    # === COMBINED: Best quality attempt ===
    {
        "name": "quality_optimized",
        "GLM_TTS_FLOW_STEPS": "10",
        "GLM_TTS_SAMPLING": "10",
        "GLM_TTS_CFG_RATE": "0.7",
        "GLM_TTS_QUANTIZATION": "none",
        "GLM_TTS_ATTENTION": "flash_attention_2",
        "GLM_TTS_VLLM_QUANTIZATION": "none",
        "GLM_TTS_FLOW_DTYPE": "fp32",
        "GLM_TTS_COMPILE_FLOW": "true",
        "GLM_TTS_COMPILE_VOCODER": "true",
    },
]

QUICK_PARAM_SETS = [
    {
        "name": "baseline",
        "GLM_TTS_FLOW_STEPS": "3",
        "GLM_TTS_SAMPLING": "2",
        "GLM_TTS_CFG_RATE": "0.7",
        "GLM_TTS_QUANTIZATION": "4bit",
        "GLM_TTS_ATTENTION": "flash_attention_2",
        "GLM_TTS_VLLM_QUANTIZATION": "fp8",
        "GLM_TTS_FLOW_DTYPE": "fp16",
        "GLM_TTS_COMPILE_FLOW": "true",
        "GLM_TTS_COMPILE_VOCODER": "true",
    },
    {
        "name": "cfg_disabled",
        "GLM_TTS_FLOW_STEPS": "3",
        "GLM_TTS_SAMPLING": "2",
        "GLM_TTS_CFG_RATE": "0.0",
        "GLM_TTS_QUANTIZATION": "4bit",
        "GLM_TTS_ATTENTION": "flash_attention_2",
        "GLM_TTS_VLLM_QUANTIZATION": "fp8",
        "GLM_TTS_FLOW_DTYPE": "fp16",
        "GLM_TTS_COMPILE_FLOW": "true",
        "GLM_TTS_COMPILE_VOCODER": "true",
    },
    {
        "name": "speed_optimized",
        "GLM_TTS_FLOW_STEPS": "1",
        "GLM_TTS_SAMPLING": "1",
        "GLM_TTS_CFG_RATE": "0.0",
        "GLM_TTS_QUANTIZATION": "4bit",
        "GLM_TTS_ATTENTION": "flash_attention_2",
        "GLM_TTS_VLLM_QUANTIZATION": "fp8",
        "GLM_TTS_FLOW_DTYPE": "fp16",
        "GLM_TTS_COMPILE_FLOW": "true",
        "GLM_TTS_COMPILE_VOCODER": "true",
    },
    {
        "name": "quality_optimized",
        "GLM_TTS_FLOW_STEPS": "10",
        "GLM_TTS_SAMPLING": "10",
        "GLM_TTS_CFG_RATE": "0.7",
        "GLM_TTS_QUANTIZATION": "none",
        "GLM_TTS_ATTENTION": "flash_attention_2",
        "GLM_TTS_VLLM_QUANTIZATION": "none",
        "GLM_TTS_FLOW_DTYPE": "fp32",
        "GLM_TTS_COMPILE_FLOW": "true",
        "GLM_TTS_COMPILE_VOCODER": "true",
    },
]


# Parameter space for random search
PARAM_SPACE = {
    "GLM_TTS_FLOW_STEPS": ["1", "2", "3", "5", "8", "10", "15"],
    "GLM_TTS_SAMPLING": ["1", "2", "5", "10", "15", "25"],
    "GLM_TTS_CFG_RATE": ["0.0", "0.3", "0.5", "0.7", "1.0"],
    "GLM_TTS_QUANTIZATION": ["4bit", "8bit", "none"],
    "GLM_TTS_ATTENTION": ["flash_attention_2", "sdpa", "eager"],
    "GLM_TTS_VLLM_QUANTIZATION": ["fp8", "none"],
    "GLM_TTS_FLOW_DTYPE": ["fp16", "fp32"],
    "GLM_TTS_COMPILE_FLOW": ["true", "false"],
    "GLM_TTS_COMPILE_VOCODER": ["true", "false"],
}

# Baseline config for focused sweeps (OPTIMAL from parameter sweep)
# Best balanced: 0.86x speed, 90% accuracy
BASELINE_CONFIG = {
    "GLM_TTS_FLOW_STEPS": "10",
    "GLM_TTS_SAMPLING": "5",
    "GLM_TTS_CFG_RATE": "0.7",
    "GLM_TTS_QUANTIZATION": "4bit",
    "GLM_TTS_ATTENTION": "flash_attention_2",
    "GLM_TTS_VLLM_QUANTIZATION": "none",
    "GLM_TTS_FLOW_DTYPE": "fp32",
    "GLM_TTS_COMPILE_FLOW": "true",
    "GLM_TTS_COMPILE_VOCODER": "true",
}


def generate_random_configs(n: int, seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """Generate N random configurations from the parameter space."""
    import random
    if seed is not None:
        random.seed(seed)
    
    configs = []
    seen = set()
    
    while len(configs) < n:
        config = {param: random.choice(values) for param, values in PARAM_SPACE.items()}
        config_key = tuple(sorted(config.items()))
        
        if config_key not in seen:
            seen.add(config_key)
            # Generate descriptive name
            name = f"rand_{len(configs)+1}_fs{config['GLM_TTS_FLOW_STEPS']}_s{config['GLM_TTS_SAMPLING']}_cfg{config['GLM_TTS_CFG_RATE']}"
            config["name"] = name
            configs.append(config)
    
    return configs


def generate_focused_configs(focus_param: str) -> List[Dict[str, Any]]:
    """Generate configs that vary only one parameter, keeping others at baseline."""
    if focus_param not in PARAM_SPACE:
        available = ", ".join(PARAM_SPACE.keys())
        raise ValueError(f"Unknown parameter: {focus_param}. Available: {available}")
    
    configs = []
    for value in PARAM_SPACE[focus_param]:
        config = BASELINE_CONFIG.copy()
        config[focus_param] = value
        
        # Generate name from the varied parameter
        short_name = focus_param.replace("GLM_TTS_", "").lower()
        config["name"] = f"focus_{short_name}_{value}"
        configs.append(config)
    
    return configs


def load_completed_runs() -> set:
    """Load names of already-completed runs to support resume."""
    completed = set()
    results_path = OUTPUT_DIR / "sweep_results_partial.json"
    if results_path.exists():
        try:
            with open(results_path) as f:
                results = json.load(f)
            for r in results:
                if r.get("speed", 0) > 0 or r.get("accuracy", 0) > 0:
                    completed.add(r.get("run_name"))
            log(f"Found {len(completed)} completed runs to skip")
        except:
            pass
    return completed


def main():
    parser = argparse.ArgumentParser(
        description="TTS Parameter Sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full exhaustive sweep (20 configs, ~5 hours)
  ./scripts/run_param_sweep.py

  # Quick sweep (4 configs, ~1 hour)  
  ./scripts/run_param_sweep.py --quick

  # Random search: test 10 random configurations
  ./scripts/run_param_sweep.py --random 10

  # Focus on one parameter: test all values of FLOW_STEPS
  ./scripts/run_param_sweep.py --focus GLM_TTS_FLOW_STEPS

  # Resume interrupted sweep
  ./scripts/run_param_sweep.py --resume
        """
    )
    parser.add_argument("--quick", action="store_true", help="Run quick sweep (4 configs)")
    parser.add_argument("--config", type=str, help="Path to custom sweep config JSON")
    parser.add_argument("--random", type=int, metavar="N", help="Random search with N configurations")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--focus", type=str, metavar="PARAM", help="Focus on one parameter (e.g. GLM_TTS_FLOW_STEPS)")
    parser.add_argument("--resume", action="store_true", help="Skip already-completed configurations")
    args = parser.parse_args()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Determine which param sets to use
    if args.random:
        param_sets = generate_random_configs(args.random, seed=args.seed)
        mode = f"Random Search ({args.random} configs)"
    elif args.focus:
        param_sets = generate_focused_configs(args.focus)
        mode = f"Focused Sweep ({args.focus})"
    elif args.config:
        with open(args.config) as f:
            param_sets = json.load(f)
        mode = f"Custom Config ({args.config})"
    elif args.quick:
        param_sets = QUICK_PARAM_SETS.copy()
        mode = "Quick Sweep"
    else:
        param_sets = DEFAULT_PARAM_SETS.copy()
        mode = "Full Exhaustive Sweep"
    
    # Deep copy param sets to avoid mutation issues
    param_sets = [dict(p) for p in param_sets]
    
    # Filter out completed runs if resuming
    if args.resume:
        completed = load_completed_runs()
        original_count = len(param_sets)
        param_sets = [p for p in param_sets if p.get("name") not in completed]
        if len(param_sets) < original_count:
            log(f"Resuming: skipping {original_count - len(param_sets)} already-completed configs")
    
    log("=" * 60)
    log("  TTS PARAMETER SWEEP")
    log("=" * 60)
    log(f"Mode: {mode}")
    log(f"Configurations: {len(param_sets)}")
    log(f"Estimated time: {len(param_sets) * 12} - {len(param_sets) * 18} minutes")
    log(f"Output: {OUTPUT_DIR}")
    
    if not param_sets:
        log("No configurations to test!")
        return 0
    
    try:
        results = run_sweep(param_sets, quick=args.quick)
        
        # Save final results
        results_path = OUTPUT_DIR / "sweep_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        log(f"\nResults saved to: {results_path}")
        
        # Generate report
        report_path = OUTPUT_DIR / "sweep_report.txt"
        generate_report(results, report_path)
        log(f"Report saved to: {report_path}")
        
    except KeyboardInterrupt:
        log("\n⚠️ Sweep interrupted by user")
        log("Use --resume to continue from where you left off")
        return 1
    except Exception as e:
        log(f"\n❌ Sweep failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

