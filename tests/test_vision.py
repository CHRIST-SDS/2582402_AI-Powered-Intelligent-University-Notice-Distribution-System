from src.vision.vision_model import LocalVisionModel


def main():

    model = LocalVisionModel()

    image_path = r"C:\Python_project\Generative AI\Christ_University_Posters\christ_poster_1.jpg"

    result = model.extract_content(image_path)

    print("\n========== GEMMA 3 VISION OUTPUT ==========\n")
    print(result)


if __name__ == "__main__":
    main()