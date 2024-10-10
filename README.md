# CANProtocolMLScanner

**Creare Sistemas** presents **CANProtocolMLScanner**, a sophisticated tool for analyzing and detecting key communication signals within vehicles using the **CAN (Controller Area Network)** protocol. By leveraging advanced machine learning algorithms, this tool is designed to identify and monitor critical features such as odometer readings, RPM, and dashboard communications.  

## Supported Features and Signals

Currently, the software has support for **Square Waves**, making it capable of identifying signals related to:

- Brake
- Windshield wipers
- Accelerator pedal
- Brake pedal

The tool can process **SPY files** (CSV format) that contain thousands of records, each separated by **PGN** (Parameter Group Number) and their associated 8 bytes.

## Supported Hardware Devices

CANProtocolMLScanner supports the following hardware devices for capturing CAN data:

- **ixxat USB-to-CAN v2 compact**
- **Simple CAN**

You can import CAN data through the **"Carregar ixxt csv"** button on the left side of the software interface.

## Interface Overview

- **Dropdown Menu (Top Right):** This menu allows you to review previous reverse engineering sessions and provides a repository of DataFrames that can be used for further training. You can select previously processed files for analysis.
  
- **Model Inference:** Use the **"Usar Modelo"** button to perform inference. The model will search for patterns in the loaded data that match the training dataset’s time-series sequences.
  
- **Plot Results:** Once the inference is complete, the algorithm can return anywhere from zero to more than 30 graphs. These results indicate how similar the CAN sequences are to the patterns from the training set. You can navigate through the graphs using the buttons located on the bottom-left corner of the interface.

## Output and Saving Results

When a **PGN** and **byte** combination is identified, you’ll have the option to save the results. It’s recommended to follow this naming format:

[CAR_MODEL]-[YEAR][EVENT][REPETITIONS][PGN][BYTE].csv


This ensures the files are well-organized for future analysis. Save the files in the **Can Data Chunks** directory using the button on the bottom-right corner.

## Preprocessing Overview

Before inference begins, the CANProtocolMLScanner preprocesses the raw data by separating the **PGNs** and their 8-byte values. For each PGN, specific bytes are extracted, and **FF** (255 in decimal) values are removed as they represent null or placeholder data. The preprocessing step ensures that only valid signal data is passed through for machine learning inference, increasing the accuracy and relevance of the results.

The system is designed to handle **Square Wave signals**, and it compares these to pre-trained LSTM (Long Short-Term Memory) models to find matching time-series patterns. This preprocessing is crucial for reducing noise and enhancing the reliability of the signal detection process.

## Instructions for Use

1. **Import CAN Data:** Use the "Carregar ixxt csv" button to load data captured from supported devices.
2. **Select File from Dropdown:** Review previously saved CSV files via the dropdown on the top right.
3. **Run Model Inference:** Click "Usar Modelo" to search for time-series patterns that match your training dataset.
4. **Plot and Review Results:** The model may produce multiple graphs based on the matching patterns found. Use the bottom-left navigation buttons to scroll through them.
5. **Save Results:** Once a significant PGN and byte are identified, save the results with a recommended file format in the Can Data Chunks directory.

## Additional Features

- **Generate PDF:** You can generate a PDF file containing all the plotted graphs. This option is available via the "Gerar PDF" button on the bottom navigation.
- **Save Raw Data:** Easily save the raw data captured for further analysis by clicking the "Salvar Dados Brutos" button on the bottom-right of the interface. You'll be prompted to specify a file name and location.

---

### Developed by

**Allan Rodrigues**
