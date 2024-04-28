Welcome to our Sports Analysis Repository, where we leverage cutting-edge Computer Vision Techniques and Deep Learning to delve into the intricacies of various sports.

1. **Download Models:** Obtain the necessary models for your specific sports analysis task from our [Model Repository](https://drive.google.com/drive/folders/1aL6ymtOGY3wFMvfnOFR2U5879onsMd1y?usp=sharing).

2. **Download Input Videos(Optional):**: Download input videos from this [link](https://drive.google.com/drive/folders/1VxnGADPKjf8Y6eEEw4FxUGwChvtNUW1s?usp=sharing)

3. **Requirements:** Install necessary dependencies
    ```bash
    pip install -r requirements.txt
    ``` 

4. **Perform Video Inference:** Utilize inference tool by executing the following command in your terminal:

    ```bash
    python main.py --sport=sport_type --model_path=model_path --inp_f_path=input_file_path
    ```

    Replace `hockey` with your desired sport and provide the appropriate paths for `model_path` and `input_file_path`.

   **Supported Sports:**'hockey', 'volleyball', 'cricket'

### For Cricket:
*For cricket video inference, append these extra arguments to your command:*

```bash
--ball_track_model
