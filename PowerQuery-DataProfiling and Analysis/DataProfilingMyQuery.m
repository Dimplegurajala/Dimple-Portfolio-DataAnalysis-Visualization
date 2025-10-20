let
    Source = Excel.Workbook(File.Contents("C:\Users\reshm\OneDrive\Desktop\Linkedin\Data Analytics- course\Exercise Files\04_04_Tips on prep\DataforTemplate.xlsx"), null, true),
    SalesOrders_Sheet = Source{[Item="SalesOrders",Kind="Sheet"]}[Data],
    #"Promoted Headers" = Table.PromoteHeaders(SalesOrders_Sheet, [PromoteAllScalars=true]),
    #"Changed Type" = Table.TransformColumnTypes(#"Promoted Headers",{{"SalesOrderID", Int64.Type}, {"OrderDate", type datetime}, {"DueDate", type datetime}, {"ShipDate", type datetime}, {"SalesOrderNumber", type text}, {"AccountNumber", type text}, {"CustomerID", Int64.Type}, {"TerritoryID", Int64.Type}, {"SubTotal", type number}, {"TaxAmt", type number}, {"Freight", type number}, {"TotalDue", type number}, {"Comment", type text}}),
    #"Changed Type1" = Table.TransformColumnTypes(#"Changed Type",{{"SalesOrderID", type text}, {"OrderDate", type date}, {"DueDate", type date}, {"ShipDate", type date}}),
    #"Replaced Value" = Table.ReplaceValue(#"Changed Type1","SO","",Replacer.ReplaceText,{"SalesOrderNumber"}),
    #"Renamed Columns" = Table.RenameColumns(#"Replaced Value",{{"AccountNumber", "MainAcctGL"}}),
    #"Duplicated Column" = Table.DuplicateColumn(#"Renamed Columns", "MainAcctGL", "MainAcctGL - Copy"),
    #"Split Column by Delimiter" = Table.SplitColumn(#"Duplicated Column", "MainAcctGL - Copy", Splitter.SplitTextByDelimiter("-", QuoteStyle.Csv), {"MainAcctGL - Copy.1", "MainAcctGL - Copy.2", "MainAcctGL - Copy.3"}),
    #"Changed Type2" = Table.TransformColumnTypes(#"Split Column by Delimiter",{{"MainAcctGL - Copy.1", Int64.Type}, {"MainAcctGL - Copy.2", Int64.Type}, {"MainAcctGL - Copy.3", Int64.Type}}),
    #"Renamed Columns1" = Table.RenameColumns(#"Changed Type2",{{"MainAcctGL - Copy.1", "GL Number"}, {"MainAcctGL - Copy.2", "Acct Number"}, {"MainAcctGL - Copy.3", "Category"}}),
    #"Removed Columns" = Table.RemoveColumns(#"Renamed Columns1",{"TerritoryID", "Comment"}),
    #"Removed Other Columns" = Table.SelectColumns(#"Removed Columns",{"SalesOrderID", "OrderDate", "DueDate", "ShipDate", "SalesOrderNumber", "MainAcctGL", "CustomerID", "SubTotal", "TaxAmt", "Freight", "TotalDue", "GL Number", "Acct Number", "Category"})
in
    #"Removed Other Columns"
