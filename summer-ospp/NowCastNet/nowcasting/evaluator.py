import mindspore as ms
from nowcasting.data_provider import datasets_factory
from nowcasting.layers.utils import plt_img
from nowcasting.evaluations import evaluate_all_metrics
import numpy as np

def infer(model, configs, calculate_metrics=True):
    test_dataset_generator = datasets_factory.RadarData(configs, module_name="generation")
    test_dataset = datasets_factory.NowcastDataset(dataset_generator=test_dataset_generator, module_name="generation")
    test_dataset = test_dataset.create_dataset(configs.batch_size)
    
    noise_scale = configs.noise_scale
    batch_size = configs.batch_size
    w_size = configs.img_width
    h_size = configs.img_height
    ngf = configs.ngf
    steps = 1
    plt_idx = [x // 9 for x in [10, 60, 120]]
    
    # Store metrics for all batches
    all_metrics = []    
    for data in test_dataset.create_dict_iterator():
        inp, evo_result, labels = data.get("inputs"), data.get("evo"), data.get("labels")
        noise = ms.tensor(ms.numpy.randn((batch_size, ngf, h_size // noise_scale, w_size // noise_scale)), inp.dtype)
        pred = model.network.gen_net(inp, evo_result, noise)
        
        for j in range(batch_size):
            pred_np = pred[j].asnumpy()
            label_np = labels[j].asnumpy()
            evo_np = evo_result[j].asnumpy() * 128
            
            # Calculate evaluation metrics if requested
            batch_metrics = None
            if calculate_metrics:
                try:
                    # Calculate metrics for this sample
                    batch_metrics = evaluate_all_metrics(
                        pred=pred_np, 
                        target=label_np,
                        pool_size=None
                    )
                    all_metrics.append(batch_metrics)
                    
                    print(f"Batch {steps}, Sample {j} Metrics:")
                    for key, value in batch_metrics.items():
                        print(f"  {key}: {value:.4f}")
                    
                except Exception as e:
                    print(f"Error calculating metrics for batch {steps}, sample {j}: {e}")
                    batch_metrics = {'Error': f'Metrics calculation failed: {str(e)}'}
            
            # Generate visualizations
            fig_name = f"generation_{steps}_{j}"
            # plt_img(
            #     field=pred_np,
            #     label=label_np,
            #     idx=plt_idx,
            #     fig_name=f"{fig_name}_original.jpg",
            #     evo=evo_np
            #     )
        
        steps += 1
    
    # Calculate and print average metrics
    if calculate_metrics and all_metrics:
        print("\n" + "="*50)
        print("OVERALL EVALUATION RESULTS")
        print("="*50)
        
        avg_metrics = {}
        for key in all_metrics[0].keys():
            if isinstance(all_metrics[0][key], (int, float)):
                values = [m[key] for m in all_metrics if isinstance(m[key], (int, float))]
                if values:
                    avg_metrics[key] = np.mean(values)
        
        print("Average Metrics across all samples:")
        for key, value in avg_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Save metrics to file with both detailed and average results
        metrics_text = "NowCast Model Evaluation Results\n"
        metrics_text += "="*60 + "\n\n"
        
        # Save detailed results for each batch
        metrics_text += "DETAILED RESULTS BY BATCH:\n"
        metrics_text += "-"*40 + "\n"
        for i, batch_metrics in enumerate(all_metrics):
            metrics_text += f"\nBatch {i+1}:\n"
            for key, value in batch_metrics.items():
                if isinstance(value, (int, float)):
                    metrics_text += f"  {key}: {value:.4f}\n"
        
        metrics_text += "\n" + "="*60 + "\n"
        metrics_text += "AVERAGE RESULTS ACROSS ALL BATCHES:\n"
        metrics_text += "-"*40 + "\n"
        for key, value in avg_metrics.items():
            metrics_text += f"{key}: {value:.4f}\n"
        
        # Additional statistics
        metrics_text += "\n" + "="*60 + "\n"
        metrics_text += "STATISTICAL SUMMARY:\n"
        metrics_text += "-"*40 + "\n"
        metrics_text += f"Total number of samples evaluated: {len(all_metrics)}\n"
        
        # Calculate standard deviations for key metrics
        key_metrics = ['MAE', 'MSE', 'RMSE']
        for metric in key_metrics:
            if metric in avg_metrics:
                values = [m[metric] for m in all_metrics if isinstance(m.get(metric), (int, float))]
                if len(values) > 1:
                    std_dev = np.std(values)
                    metrics_text += f"{metric} - Mean: {avg_metrics[metric]:.4f}, Std: {std_dev:.4f}\n"
        
        # Save to file
        with open("evaluation_results.txt", "w", encoding='utf-8') as f:
            f.write(metrics_text)
        
        print(f"\nDetailed metrics saved to: evaluation_results.txt")
        print(f"Saved detailed results for {len(all_metrics)} batches")
    
    print(f"\nInference completed. Generated {steps-1} batches of results.")