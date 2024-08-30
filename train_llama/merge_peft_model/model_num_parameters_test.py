from transformers import AutoModel

model_name = "./models/w_6_stage1"
model = AutoModel.from_pretrained(model_name)

num_parameters = model.num_parameters()
print(f"The model has {num_parameters} parameters.")
