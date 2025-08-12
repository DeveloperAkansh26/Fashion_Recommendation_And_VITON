import os
import json
import time
from PIL import Image
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, TimeoutError


# os.environ["GEMINI_API_KEY"] = "AIzaSyAWSBHvaf8NYPUdtyslgrYjo9pl-Sw_dIk"      #My
# os.environ["GEMINI_API_KEY"] = "AIzaSyDryzLjl84tQcdIqdXA_RwI2-KXGQMh4M0"      #My2
# os.environ["GEMINI_API_KEY"] = "AIzaSyDNWii5DoKOjnBdci9BOc-92pb0HtyyDpM"      #My3
# os.environ["GEMINI_API_KEY"] = "AIzaSyCJdmdLoPpPQ96cVpYaPBLtFh_RVza_iHo"      #My4
# os.environ["GEMINI_API_KEY"] = "AIzaSyCQF7BRQRWoqwsJWq5qtNEW-hjF2wef6wo"      #My5
# os.environ["GEMINI_API_KEY"] = "AIzaSyC7G7ipXiu8zclTTNIrv9cPZY7X9GEVsek"      #My6
os.environ["GEMINI_API_KEY"] = "AIzaSyDjGpJ8Oi3EiATQNe85BjruiAlXpSwy8y4"     #DTU

# os.environ["GEMINI_API_KEY"] = "AIzaSyDnYkOL6352jbCCA1KCucmWMCpT4Jlvjcg"     #INFERNO
# os.environ["GEMINI_API_KEY"] = "AIzaSyAhT920j930ZUo0xz6AzPKKyos9lj3hmTA"     #ALT

# os.environ["GEMINI_API_KEY"] = "AIzaSyBddkpG0MiFjiKrn9J0MyOyb2qWixLCQgI"      #papa
# os.environ["GEMINI_API_KEY"] = "AIzaSyBC02TBHoqzRHUlRZd0vg2B2Y418_OH3-c"    #mummy

# os.environ["GEMINI_API_KEY"] = "AIzaSyAjzOHSY1moLprCYz0_1mejvFzvuKlVYNk"
# os.environ["GEMINI_API_KEY"] = "AIzaSyB-ktzuf7ozndzqzf-Ieo3M926yJBfdv0k"
# os.environ["GEMINI_API_KEY"] = "AIzaSyA37ffaXHfV9UQMqt5V6mfHVpFJ0vl1hhs"

# os.environ["GEMINI_API_KEY"] = "AIzaSyCAPx8e-VrHlmz4YanKeDLH_EsD1ImvPPM"
# os.environ["GEMINI_API_KEY"] = "AIzaSyDC3cUOXlMhGrkMMOJ3f1KIegRL3DH6CQ0"

# os.environ["GEMINI_API_KEY"] = "AIzaSyDL7O7XylySyYBUbuaamPscRXcUyz9gXgE"    #Himani
# os.environ["GEMINI_API_KEY"] = "AIzaSyBO9-Dp16d3tmX4HjKd7zaS4DN-i6Fh7WQ"    #Himani

# os.environ["GEMINI_API_KEY"] = "AIzaSyA07V3UOuRCG2iBYFXOXS9VzZaw5dPpZRk"  # Sol T.


genai.configure(api_key=os.environ["GEMINI_API_KEY"])
gemini = genai.GenerativeModel("gemini-2.0-flash")


def gemini_api_request(prompt, image, timeout=25):
    def make_request():

        response = gemini.generate_content([image, prompt])
        return response

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(make_request)
        # try:
        return future.result(timeout=timeout)
        # except TimeoutError:
        #     print("Request timed out.")
        #     return None
        

IMAGE_FOLDER = "clothes_tryon_dataset/train/cloth"
OUTPUT_PROMPTS_FILE = "generated_prompts.json"


print(f"Starting prompt generation for images in: {IMAGE_FOLDER}")
print(f"Prompts will be saved to: {OUTPUT_PROMPTS_FILE}")

image_files = sorted([f for f in os.listdir(IMAGE_FOLDER)])
image_files.sort(key=lambda f: int(f.split("_")[0]))

generated_prompts = {}
gemini_prompt_template = "Describe the image in short. Include only necessary details about the cloth that the person is wearing such that a person is able to understand the overall appearance of the cloth. Start you sentence with: A person wearing .... against a plane white background. I will use this output for text conditioning of a diffusion model so make the output appropriate."


done = 6507


try:
    for i, img_name in enumerate(image_files[done:]):

        img_path = os.path.join(IMAGE_FOLDER, img_name)
        image = Image.open(img_path)

        result = gemini.generate_content([image, gemini_prompt_template])
        prompt_text = result.text

        generated_prompts[img_name] = prompt_text
        time.sleep(2)

        if i % 10 == 0:
            print(f"Processed {i+1}/{len(image_files)} images. Last prompt for {img_name}: {prompt_text[:50]}...")

except Exception as e:
    print(f"An error occurred: {e}")
    with open(OUTPUT_PROMPTS_FILE, 'a') as f:
        json.dump(generated_prompts, f, indent=4)

    print(f"Finished generating prompts. Saved to {OUTPUT_PROMPTS_FILE}")
    print(f"Last Included Index No:", done + i-1)

except KeyboardInterrupt:

    with open(OUTPUT_PROMPTS_FILE, 'a') as f:
        json.dump(generated_prompts, f, indent=4)

    print(f"Finished generating prompts. Saved to {OUTPUT_PROMPTS_FILE}")
    print(f"Last Included Index No:", done + i-1)