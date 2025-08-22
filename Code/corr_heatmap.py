import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_corr_heatmap(data_encoded):
    # Generate a heatmap for feature correlation using encoded data
    plt.figure(figsize=(15, 12))  
    correlation_matrix = data_encoded.corr()
    sns.heatmap(
        correlation_matrix, 
        annot=True,          
        cmap="coolwarm",     
        fmt=".2f",           
        linewidths=0.5,      
        annot_kws={"size": 8}  
    )
    plt.title("Feature Correlation Heatmap", fontsize=16)  
    plt.xticks(rotation=45, fontsize=10, ha='right')  
    plt.yticks(fontsize=10)  
    plt.tight_layout()
    name = "correlation_heatmap.png"
    filepath = os.path.join("./Visuals", name)
    plt.savefig(filepath)
