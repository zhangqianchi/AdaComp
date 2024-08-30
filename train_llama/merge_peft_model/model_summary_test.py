from torchsummary import summary
from transformers import LlamaForCausalLM

# Create the fine-tuned model
model = LlamaForCausalLM.from_pretrained('./models/w_6_stage1_all')

# Use torchsummary's summary function to view the model's structure and parameter count
summary(model, input_size=(batch_size, input_size))