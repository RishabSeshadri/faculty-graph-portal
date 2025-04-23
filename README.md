# Faculty Graph Portal
The main purpose of this research project is to create a portal for University of Georgia engineering faculty to find each other based on research, expertise, etc.

----------------------------------------------
## Requirements


Ensure all the libraries are correctly installed (i.e. openai, pyvis, etc.), and Python 3 is being used. You can install these libraries by running:

```$ pip3 install requirements.txt```


You will need an **OpenAI API key** to run the code. You can get one by signing up for an account at [OpenAI](https://platform.openai.com/signup). For reference, I purchased \$5 worth of credit to use for this project in January, and with extensive testing, have used about \$0.42 over the course of 4 months. 

Once you have your API key, create a file called `.env` in the root directory of the project and add the following line to it:
    
    API_KEY=your_api_key_here

You will also need the data used for the project. You can contact me for the cleaned data or reach out to Dr. Ramasamy for the original data. The data must be converted into a CSV format to read it into the code.

Once you have the data, you can add the following line to the `.env` file:

    CSV_FILE=path/to/your/data.csv

----------------------------------------------

## Project Structure    

The project is structured as follows:

```
Faculty-Graph-Portal:
├── .env
├── .gitignore
├── README.md
├── faculty_graph.py
├── faculty_graph.ipynb
├── output
│   ├── category
|   │   ├── output_category_graphs_here.html
│   ├── faculty
|   │   ├── output_faculty_graphs_here.html
├── data
│   ├── your_csv_here.csv
├── logs
│   ├── logged_reasoning_and_data_here.log
```

The notebook serves as a testing ground for the code, while the `faculty_graph.py` file contains the main code for the project. The `output` folder contains the generated graphs, and the `logs` folder contains the logged reasoning and data.
The reasoning for choosing specified weights is noted in the `logs` folder and may be worth observing for debugging and improvement purposes.

----------------------------------------------
## Current TODOs and Future Steps
  - **Testing the code:** The code is not fully tested and may require some debugging. Some of the functionality may need to be brought up to date between the notebook and the module.
  - **Active frontend:** As it stands, the graph is generated and saved as an HTML file that allows for interactivity, but this is not the ideal functionality. It may be better to have this work as a launchable application.
  - **OpenAI Integration:** The current code uses OpenAI's API to adjust weights and create JSON files with expertise columns for better connections. There is a lot of potential paths to explore for how the data is cleaned, produced, and modeled. The core functionality here may define how efficient of an application it may be.
  - **Performance:** The API calls take time to run, anywhere around 30-45 seconds, which could be inefficient when looking at this with a live project. However, it is possible to generate these graphs in the backgroun on launch or simply once, and store the result/pull the result as needed when calls are made. This depends on the desired functionality.
  - **Data acquisition:** One of the biggest current limitations for this project is the data that is used with training. Getting more data would be extremely useful in both training and graph generation. Any more professors, research expertise, related fields, or overall standardized data would help.

## Contact

If you have any questions or suggestions, feel free to reach out to me at [rishab.seshadri@gmail.com](rishab.seshadri@gmail.com).

----------------------------------------------
*Authors: Rishab Seshadri*