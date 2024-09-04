import re
import matplotlib.pyplot as plt

def find_scores_from_file(file_path):  
    with open(file_path, 'r', encoding='utf-8') as file:  
        lines = file.readlines()  
        
    lines.reverse()
    # 定义正则表达式  
    pattern = r'(F1 score: \d+\.\d+), (Em score: \d+\.\d+), (total_count: \d+)'  
    pattern = r'F1 score: (\S+), Em score: (\S+), total_count: (\d+)'  
    for line in lines:
        # 使用正则表达式搜索文本  
        match = re.search(pattern, line)  
        if match:  
            f1_score = match.group(1)  
            em_score = match.group(2)  
            total_count = match.group(3)  
            print(f"F1 score: {f1_score}, Em score: {em_score}, total_count: {total_count}")  
            break



def parse_log_for_step_loss(log_file_path):
    """
    Parses a log file to extract step and loss information for plotting.

    Args:
        log_file_path (str): The path to the log file.

    Returns:
        list of tuples: Each tuple contains (step, loss) extracted from the log.
    """
    step_loss_pairs = []
    # Updated regular expression pattern to match the step and loss in the log
    pattern = r"step:\[\s*(\d+)/\s*\d+\], loss: ([\d.]+)"

    with open(log_file_path, 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                step = int(match.group(1).strip())
                loss = float(match.group(2))
                step_loss_pairs.append((step, loss))

    return step_loss_pairs



def plot_step_loss(step_loss_pairs,arv=10):
    """
    Plots the step vs loss data.

    Args:
        step_loss_pairs (list of tuples): Each tuple contains (step, loss).
    """
    steps, losses = zip(*step_loss_pairs)  # Unpack the step_loss_pairs into two lists
    arv_losses = []
    for i in range(len(losses)):
        if i > arv:
            sub_losses = losses[i-arv:i+arv]
            avg_loss = sum(sub_losses) / len(sub_losses)
        else:
            sub_losses = losses[0:i+arv]
            avg_loss = sum(sub_losses) / len(sub_losses)
        arv_losses.append(avg_loss)
    #arv_losses = [sum(losses[i-arv:i+arv])/len(losses[i-arv:i+arv]) if i >arv else sum(losses[0:i+arv])/len(losses[0:i+arv]) for i in range(len(losses))]
    #arv_losses = [sum(losses[i+1-arv:i+1])/arv  if i >arv else sum(losses[:i+1])/(i+1) for i in range(len(losses)) ]
    plt.figure(figsize=(10, 5))  # Create a new figure with a specific size
    plt.plot(steps, losses, marker='o')  # Plot steps vs losses
    plt.plot(steps, arv_losses)  # Plot steps vs losses
    plt.title('Step vs Loss')  # Set the title of the plot
    plt.xlabel('Step')  # Set the label for the x-axis
    plt.ylabel('Loss')  # Set the label for the y-axis
    plt.grid(True)  # Enable grid
    plt.savefig('img/step_loss_plot.png')
    plt.show()  # Display the plot
    
