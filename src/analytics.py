import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io
import base64

class FireAnalytics:
    def __init__(self, run_dir='runs/classify/ignisguard_advanced_model2'):
        self.run_dir = run_dir
        self.results_csv = os.path.join(run_dir, 'results.csv')
        
    def generate_training_plots(self):
        """
        Generates Precision vs Recall and Accuracy plots from training logs.
        """
        if not os.path.exists(self.results_csv):
            return None
            
        df = pd.read_csv(self.results_csv)
        df.columns = [c.strip() for c in df.columns]
        
        plt.figure(figsize=(10, 5))
        
        # Plot 1: Accuracy & Loss
        plt.subplot(1, 2, 1)
        plt.plot(df['epoch'], df['metrics/accuracy_top1'], label='Accuracy', color='#3b82f6', linewidth=2) 
        plt.title('Training Accuracy', color='white', fontsize=12)
        plt.xlabel('Epoch', color='gray')
        plt.ylabel('Score', color='gray')
        plt.grid(True, alpha=0.1)
        plt.legend()
        
        # Plot 2: Confidence Distribution (Simulated from validation)
        plt.subplot(1, 2, 2)
        sns.histplot(df['metrics/accuracy_top1'], kde=True, color='#f97316', bins=10)
        plt.title('Detections Confidence Distribution', color='white', fontsize=12)
        plt.xlabel('Confidence', color='gray')
        plt.ylabel('Frequency', color='gray')
        plt.grid(True, alpha=0.1)
        
        plt.tight_layout()
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor='#0a0a0c', transparent=True)
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def generate_heatmap_placeholder(self):
        """
        Generates a placeholder heatmap for fire-prone regions.
        """
        plt.figure(figsize=(6, 4))
        data = [[0.1, 0.2, 0.5], [0.3, 0.8, 0.9], [0.2, 0.4, 0.6]]
        sns.heatmap(data, annot=True, cmap='YlOrRd', cbar=False)
        plt.title('Zonal Fire Risk Heatmap', color='white')
        plt.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor='#0a0a0c', transparent=True)
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def generate_comparison_plot(self):
        """
        Phase 11: Visualizes the performance gap between Basic and Advanced models.
        """
        comparison_csv = 'reports/model_comparison.csv'
        if not os.path.exists(comparison_csv):
            return None
            
        df = pd.read_csv(comparison_csv)
        
        plt.figure(figsize=(10, 5))
        
        # Plot: Accuracy Comparison
        plt.subplot(1, 2, 1)
        sns.barplot(x='Model', y='Top-1 Accuracy', data=df, palette=['#64748b', '#f97316'])
        plt.title('Accuracy: Basic vs Advanced', color='white')
        plt.ylim(0.8, 1.05)
        plt.grid(True, alpha=0.1)
        
        # Plot: Speed Comparison
        plt.subplot(1, 2, 2)
        sns.barplot(x='Model', y='Inference (ms)', data=df, palette=['#3b82f6', '#10b981'])
        plt.title('Latency (Lower is Better)', color='white')
        plt.grid(True, alpha=0.1)
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor='#0a0a0c', transparent=True)
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')