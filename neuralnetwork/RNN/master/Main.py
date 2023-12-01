from method import train, predict, plot_predictions, predict_reverse, train_reverse

# train()
output_name, top5_each_step = predict('female', 'Ed')
print(output_name)
plot_predictions(top5_each_step)

# train_reverse()
output_name_reverse, top5_each_step_reverse = predict_reverse('male', 'en')
print(output_name_reverse)
plot_predictions(top5_each_step_reverse)
