# Power Query Cleaning Scripts
This repository provides reusable Power Query (M Code) scripts for common data cleaning and analysis tasks in Excel.

## Files Included:

.m files: The Power Query logic (M Code).

.csv files: Sample data to test and demonstrate each query.

## How to Use These Queries (2 Steps)

1. Import the Query LogicIn Excel, go to the Data tab $\rightarrow$
   
  Get Data $\rightarrow$ From Other Sources $\rightarrow$ Blank Query.

  In the Power Query Editor, go to the View tab and click Advanced Editor.

  Open the desired .m file from this repository (e.g., Clean_TrimWhitespace.m).

  Copy all the M Code and paste it into the Advanced Editor, replacing the existing content.

  Click Done.

2. Update the Source File Name
   
  The query is currently pointing to a sample file name. You need to quickly change this source to point to your data file.

  In the Power Query Editor, check the Applied Steps pane on the right and click the very first step, usually named Source.

  Look at the Formula Bar (at the top). You will see the file name inside the formula (e.g., "SampleData.csv").

  Replace the sample file name with the exact name of the file you want to load (e.g., "MyNewProject.xlsx").
