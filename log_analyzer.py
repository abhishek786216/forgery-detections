"""
Log Viewer and Analysis Utility
View, analyze, and summarize training logs
"""

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import glob
import json


class LogAnalyzer:
    """
    Analyze and visualize training logs
    """
    
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.experiments = self.find_experiments()
    
    def find_experiments(self):
        """Find all experiment directories"""
        experiments = []
        if os.path.exists(self.log_dir):
            for item in os.listdir(self.log_dir):
                exp_path = os.path.join(self.log_dir, item)
                if os.path.isdir(exp_path):
                    experiments.append({
                        'name': item,
                        'path': exp_path,
                        'files': os.listdir(exp_path)
                    })
        return experiments
    
    def list_experiments(self):
        """List all available experiments"""
        print("="*80)
        print("AVAILABLE EXPERIMENTS")
        print("="*80)
        
        if not self.experiments:
            print("No experiments found in log directory.")
            return
        
        for i, exp in enumerate(self.experiments):
            print(f"{i+1}. {exp['name']}")
            print(f"   Path: {exp['path']}")
            print(f"   Files: {', '.join(exp['files'])}")
            print("-" * 40)
    
    def view_terminal_log(self, experiment_name):
        """View terminal log for an experiment"""
        exp = self.find_experiment(experiment_name)
        if not exp:
            print(f"Experiment '{experiment_name}' not found!")
            return
        
        # Find terminal log file
        terminal_log = None
        for file in exp['files']:
            if 'terminal.log' in file:
                terminal_log = os.path.join(exp['path'], file)
                break
        
        if not terminal_log or not os.path.exists(terminal_log):
            print(f"Terminal log not found for experiment '{experiment_name}'")
            return
        
        print(f"="*80)
        print(f"TERMINAL LOG: {experiment_name}")
        print(f"="*80)
        
        with open(terminal_log, 'r', encoding='utf-8') as f:
            content = f.read()
            print(content)
    
    def view_training_log(self, experiment_name):
        """View training log for an experiment"""
        exp = self.find_experiment(experiment_name)
        if not exp:
            print(f"Experiment '{experiment_name}' not found!")
            return
        
        # Find training log file
        training_log = None
        for file in exp['files']:
            if 'training.log' in file:
                training_log = os.path.join(exp['path'], file)
                break
        
        if not training_log or not os.path.exists(training_log):
            print(f"Training log not found for experiment '{experiment_name}'")
            return
        
        print(f"="*80)
        print(f"TRAINING LOG: {experiment_name}")
        print(f"="*80)
        
        with open(training_log, 'r', encoding='utf-8') as f:
            content = f.read()
            print(content)
    
    def analyze_preprocessing(self, experiment_name):
        """Analyze preprocessing logs for an experiment"""
        exp = self.find_experiment(experiment_name)
        if not exp:
            print(f"Experiment '{experiment_name}' not found!")
            return
        
        # Find preprocessing stats CSV
        stats_csv = None
        for file in exp['files']:
            if 'preprocessing_stats.csv' in file:
                stats_csv = os.path.join(exp['path'], file)
                break
        
        if not stats_csv or not os.path.exists(stats_csv):
            print(f"Preprocessing stats CSV not found for experiment '{experiment_name}'")
            return
        
        try:
            # Load preprocessing stats
            # Check if header exists, if not add it
            with open(stats_csv, 'r') as f:
                first_line = f.readline().strip()
            
            if first_line.startswith('timestamp,'):
                df = pd.read_csv(stats_csv)
            else:
                # Add header manually
                columns = ['timestamp', 'operation', 'file_name', 'file_size_kb', 'processing_time_ms', 'success', 'error_message']
                df = pd.read_csv(stats_csv, names=columns)
            
            print(f"="*80)
            print(f"PREPROCESSING ANALYSIS: {experiment_name}")
            print(f"="*80)
            
            # Basic statistics
            print("PROCESSING SUMMARY:")
            print("-" * 40)
            print(f"Total file operations: {len(df)}")
            
            success_count = (df['success'] == True).sum()
            failure_count = (df['success'] == False).sum()
            success_rate = success_count / len(df) * 100
            
            print(f"Successful operations: {success_count}")
            print(f"Failed operations: {failure_count}")
            print(f"Success rate: {success_rate:.1f}%")
            
            # Processing times
            print(f"\nPROCESSING TIMES:")
            print("-" * 40)
            avg_time = df['processing_time_ms'].mean()
            min_time = df['processing_time_ms'].min()
            max_time = df['processing_time_ms'].max()
            
            print(f"Average processing time: {avg_time:.2f} ms")
            print(f"Fastest operation: {min_time:.2f} ms")
            print(f"Slowest operation: {max_time:.2f} ms")
            
            # By operation type
            print(f"\nBY OPERATION TYPE:")
            print("-" * 40)
            for operation in df['operation'].unique():
                op_df = df[df['operation'] == operation]
                op_success_rate = (op_df['success'] == True).sum() / len(op_df) * 100
                avg_op_time = op_df['processing_time_ms'].mean()
                print(f"{operation}: {len(op_df)} files, {op_success_rate:.1f}% success, {avg_op_time:.2f}ms avg")
            
            # File sizes
            print(f"\nFILE SIZES:")
            print("-" * 40)
            avg_size = df['file_size_kb'].mean()
            min_size = df['file_size_kb'].min()
            max_size = df['file_size_kb'].max()
            
            print(f"Average file size: {avg_size:.1f} KB")
            print(f"Smallest file: {min_size:.1f} KB")
            print(f"Largest file: {max_size:.1f} KB")
            
            # Errors if any
            error_df = df[df['success'] == False]
            if not error_df.empty:
                print(f"\nERRORS:")
                print("-" * 40)
                error_counts = error_df['error_message'].value_counts()
                for error, count in error_counts.head(5).items():
                    print(f"{error}: {count} occurrences")
            
            # Plot preprocessing statistics
            self.plot_preprocessing_stats(df, experiment_name)
            
        except Exception as e:
            print(f"Error analyzing preprocessing: {e}")
    
    def plot_preprocessing_stats(self, df, experiment_name):
        """Plot preprocessing statistics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Preprocessing Statistics: {experiment_name}', fontsize=16)
        
        # Processing time histogram
        axes[0, 0].hist(df['processing_time_ms'], bins=30, alpha=0.7, color='blue')
        axes[0, 0].set_title('Processing Time Distribution')
        axes[0, 0].set_xlabel('Processing Time (ms)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # File size histogram
        axes[0, 1].hist(df['file_size_kb'], bins=30, alpha=0.7, color='green')
        axes[0, 1].set_title('File Size Distribution')
        axes[0, 1].set_xlabel('File Size (KB)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Success/failure by operation
        operation_counts = df.groupby(['operation', 'success']).size().unstack(fill_value=0)
        operation_counts.plot(kind='bar', ax=axes[1, 0], color=['red', 'green'])
        axes[1, 0].set_title('Success/Failure by Operation')
        axes[1, 0].set_xlabel('Operation Type')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].legend(['Failed', 'Success'])
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Processing time by operation
        df.boxplot(column='processing_time_ms', by='operation', ax=axes[1, 1])
        axes[1, 1].set_title('Processing Time by Operation')
        axes[1, 1].set_xlabel('Operation Type')
        axes[1, 1].set_ylabel('Processing Time (ms)')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.log_dir, f"{experiment_name}_preprocessing_plot.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Preprocessing plot saved to: {plot_path}")
        
        plt.show()
    
    def analyze_metrics(self, experiment_name):
        """Analyze metrics CSV for an experiment"""
        exp = self.find_experiment(experiment_name)
        if not exp:
            print(f"Experiment '{experiment_name}' not found!")
            return
        
        # Check if this is a preprocessing experiment
        preprocessing_files = [f for f in exp['files'] if 'preprocessing' in f]
        if preprocessing_files:
            print("This appears to be a preprocessing experiment.")
            print("Use --action analyze_preprocessing instead.")
            return
        
        # Find metrics CSV file
        metrics_csv = None
        for file in exp['files']:
            if 'metrics.csv' in file:
                metrics_csv = os.path.join(exp['path'], file)
                break
        
        if not metrics_csv or not os.path.exists(metrics_csv):
            print(f"Metrics CSV not found for experiment '{experiment_name}'")
            return
        
        try:
            # Load metrics data
            df = pd.read_csv(metrics_csv)
            
            print(f"="*80)
            print(f"METRICS ANALYSIS: {experiment_name}")
            print(f"="*80)
            
            # Basic statistics
            print("TRAINING SUMMARY:")
            print("-" * 40)
            
            # Separate train and validation data
            train_data = df[df['split'] == 'train']
            val_data = df[df['split'] == 'val']
            
            if not train_data.empty:
                print(f"Training epochs: {len(train_data)}")
                print(f"Best training IoU: {train_data['iou'].max():.4f}")
                print(f"Best training F1: {train_data['f1_score'].max():.4f}")
                print(f"Final training loss: {train_data['loss'].iloc[-1]:.6f}")
            
            if not val_data.empty:
                print(f"Validation epochs: {len(val_data)}")
                print(f"Best validation IoU: {val_data['iou'].max():.4f}")
                print(f"Best validation F1: {val_data['f1_score'].max():.4f}")
                print(f"Final validation loss: {val_data['loss'].iloc[-1]:.6f}")
            
            # Plot metrics
            self.plot_metrics(df, experiment_name)
            
        except Exception as e:
            print(f"Error analyzing metrics: {e}")
    
    def plot_metrics(self, df, experiment_name):
        """Plot training metrics"""
        # Separate train and validation data
        train_data = df[df['split'] == 'train']
        val_data = df[df['split'] == 'val']
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Metrics: {experiment_name}', fontsize=16)
        
        # Loss plot
        if not train_data.empty:
            axes[0, 0].plot(train_data['epoch'], train_data['loss'], label='Train Loss', color='blue')
        if not val_data.empty:
            axes[0, 0].plot(val_data['epoch'], val_data['loss'], label='Val Loss', color='red')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # IoU plot
        if not train_data.empty:
            axes[0, 1].plot(train_data['epoch'], train_data['iou'], label='Train IoU', color='blue')
        if not val_data.empty:
            axes[0, 1].plot(val_data['epoch'], val_data['iou'], label='Val IoU', color='red')
        axes[0, 1].set_title('IoU Score')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('IoU')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 Score plot
        if not train_data.empty:
            axes[1, 0].plot(train_data['epoch'], train_data['f1_score'], label='Train F1', color='blue')
        if not val_data.empty:
            axes[1, 0].plot(val_data['epoch'], val_data['f1_score'], label='Val F1', color='red')
        axes[1, 0].set_title('F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Pixel Accuracy plot
        if not train_data.empty:
            axes[1, 1].plot(train_data['epoch'], train_data['pixel_accuracy'], label='Train Acc', color='blue')
        if not val_data.empty:
            axes[1, 1].plot(val_data['epoch'], val_data['pixel_accuracy'], label='Val Acc', color='red')
        axes[1, 1].set_title('Pixel Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.log_dir, f"{experiment_name}_metrics_plot.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Metrics plot saved to: {plot_path}")
        
        plt.show()
    
    def find_experiment(self, experiment_name):
        """Find experiment by name"""
        for exp in self.experiments:
            if exp['name'] == experiment_name:
                return exp
        return None
    
    def generate_report(self, experiment_name):
        """Generate comprehensive report for an experiment"""
        exp = self.find_experiment(experiment_name)
        if not exp:
            print(f"Experiment '{experiment_name}' not found!")
            return
        
        report_path = os.path.join(exp['path'], f"{experiment_name}_report.txt")
        
        with open(report_path, 'w') as f:
            f.write(f"TRAINING REPORT: {experiment_name}\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write("="*80 + "\n\n")
            
            # Experiment info
            f.write("EXPERIMENT INFORMATION:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Name: {experiment_name}\n")
            f.write(f"Path: {exp['path']}\n")
            f.write(f"Files: {', '.join(exp['files'])}\n\n")
            
            # Analyze metrics if available
            metrics_csv = None
            for file in exp['files']:
                if 'metrics.csv' in file:
                    metrics_csv = os.path.join(exp['path'], file)
                    break
            
            if metrics_csv and os.path.exists(metrics_csv):
                try:
                    df = pd.read_csv(metrics_csv)
                    
                    f.write("TRAINING METRICS SUMMARY:\n")
                    f.write("-" * 40 + "\n")
                    
                    train_data = df[df['split'] == 'train']
                    val_data = df[df['split'] == 'val']
                    
                    if not train_data.empty:
                        f.write(f"Training epochs: {len(train_data)}\n")
                        f.write(f"Best training IoU: {train_data['iou'].max():.4f}\n")
                        f.write(f"Best training F1: {train_data['f1_score'].max():.4f}\n")
                        f.write(f"Final training loss: {train_data['loss'].iloc[-1]:.6f}\n")
                    
                    if not val_data.empty:
                        f.write(f"Validation epochs: {len(val_data)}\n")
                        f.write(f"Best validation IoU: {val_data['iou'].max():.4f}\n")
                        f.write(f"Best validation F1: {val_data['f1_score'].max():.4f}\n")
                        f.write(f"Final validation loss: {val_data['loss'].iloc[-1]:.6f}\n")
                    
                except Exception as e:
                    f.write(f"Error analyzing metrics: {e}\n")
            
            f.write("\n" + "="*80)
        
        print(f"Report generated: {report_path}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Log Viewer and Analysis Utility')
    
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Log directory')
    parser.add_argument('--action', type=str, required=True,
                       choices=['list', 'view_terminal', 'view_training', 'analyze', 'analyze_preprocessing', 'report'],
                       help='Action to perform')
    parser.add_argument('--experiment', type=str,
                       help='Experiment name (required for view/analyze actions)')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    analyzer = LogAnalyzer(args.log_dir)
    
    if args.action == 'list':
        analyzer.list_experiments()
    
    elif args.action == 'view_terminal':
        if not args.experiment:
            print("Error: --experiment required for view_terminal action")
            return
        analyzer.view_terminal_log(args.experiment)
    
    elif args.action == 'view_training':
        if not args.experiment:
            print("Error: --experiment required for view_training action")
            return
        analyzer.view_training_log(args.experiment)
    
    elif args.action == 'analyze':
        if not args.experiment:
            print("Error: --experiment required for analyze action")
            return
        analyzer.analyze_metrics(args.experiment)
    
    elif args.action == 'analyze_preprocessing':
        if not args.experiment:
            print("Error: --experiment required for analyze_preprocessing action")
            return
        analyzer.analyze_preprocessing(args.experiment)
    
    elif args.action == 'report':
        if not args.experiment:
            print("Error: --experiment required for report action")
            return
        analyzer.generate_report(args.experiment)


if __name__ == '__main__':
    main()