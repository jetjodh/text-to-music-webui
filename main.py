from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy

class MusicGenerator:
    def __init__(self, model_name):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = MusicgenForConditionalGeneration.from_pretrained(model_name)

    def generate(self, text):
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        outputs = self.model.generate(**inputs)
        return self.processor.batch_decode(outputs, skip_special_tokens=True, max_new_tokens=256)
    
test = MusicGenerator("facebook/musicgen-small")
audio_values = test.generate("Hello world")
sampling_rate = test.model.config.audio_encoder.sampling_rate
scipy.io.wavfile.write("musicgen_out.wav", rate=sampling_rate, data=audio_values[0, 0].numpy())
