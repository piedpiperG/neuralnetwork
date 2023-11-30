from method import train, predict, plot_predictions

# train()
output_name, top5_each_step = predict('female', 'Ak')
print(output_name)
plot_predictions(top5_each_step)
