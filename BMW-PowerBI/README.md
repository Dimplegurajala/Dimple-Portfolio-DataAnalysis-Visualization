## Interactive BMW Used Car Sales Dashboard:

This repository contains an interactive Power BI dashboard for analyzing the sales data of used BMW cars. The project's goal was to move beyond static charts and create a dynamic tool that allows users to explore, question, and understand the key factors that drive vehicle value.

## Dashboard Preview & Key Findings
The dashboard is designed for interactive analysis. Below are two examples showcasing its capabilities and the insights uncovered during the project.

1. Analyzing Market Segments by Transmission Type
Question: How does transmission type really affect pricing across different models?

The dashboard answers this through cross-filtering. The GIF below demonstrates how clicking on the "Automatic" transmission segment in the treemap instantly updates the bar chart. This reveals the specific average price for only the automatic variants of each BMW model, providing a much deeper insight than a general average and allowing for direct price comparisons.

2. Testing Hypotheses with the Engine Size Slicer
Question: What is the relationship between engine size, fuel type, and the price-to-mileage ratio?

Using the interactive engine size slicer, I was able to drill down into the data and validate several key observations about BMW's engineering and market positioning:

Electric Models: These consistently showed the smallest engine size (around 1.0L), which is an interesting data artifact as they lack a traditional combustion engine.

Petrol Models: This category featured the largest engines, with high-performance vehicles stretching up to a powerful 6.6L.

The Versatile 3.0L Engine: This specific engine size proved to be a common feature across most non-electric fuel types, highlighting its importance and versatility in BMW's lineup.

The GIF below shows this slicer in action, demonstrating how effortlessly a user can isolate specific performance classes to analyze their unique market characteristics.


## Tools and Technologies
- Data Visualization & Analysis: Microsoft Power BI
- Dataset: Used BMW Car Sales Data (bmw.csv)

## Repository Contents
This repository includes the following files:
1) BMW.pbix: The main Power BI project file containing the data model, DAX calculations, and all visualizations.
2) bmw.csv: The raw dataset used for this analysis, containing details on over 10,000 used BMW listings.
3) BMW.pdf: A static, high-resolution export of the final dashboard for a quick overview.
4) GIFs: The animated images in this README, showcasing the dashboard's interactivity.
