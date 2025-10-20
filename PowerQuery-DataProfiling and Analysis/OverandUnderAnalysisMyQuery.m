let
    Source = Excel.Workbook(File.Contents("C:\Users\reshm\OneDrive\Desktop\Linkedin\Data Analytics- course\Exercise Files\05_01_Transform data in Excel\OverandUnderAnalysis.xlsx"), null, true),
    Suppliers_Sheet = Source{[Item="Suppliers",Kind="Sheet"]}[Data],
    #"Promoted Headers" = Table.PromoteHeaders(Suppliers_Sheet, [PromoteAllScalars=true]),
    #"Changed Type" = Table.TransformColumnTypes(#"Promoted Headers",{{"SupplierName", type text}, {"SupplierTransactionID", Int64.Type}, {"SupplierID", Int64.Type}, {"PurchaseOrderID", Int64.Type}, {"SupplierInvoiceNumber", Int64.Type}, {"TransactionDate", type date}, {"AmountExcludingTax", type number}, {"TaxAmount", type number}, {"FinalizationDate", type date}}),
    #"Uppercased Text" = Table.TransformColumns(#"Changed Type",{{"SupplierName", Text.Upper, type text}}),
    #"Duplicated Column" = Table.DuplicateColumn(#"Uppercased Text", "TransactionDate", "TransactionDate - Copy"),
    #"Extracted Year" = Table.TransformColumns(#"Duplicated Column",{{"TransactionDate - Copy", Date.Year, Int64.Type}}),
    #"Reordered Columns" = Table.ReorderColumns(#"Extracted Year",{"SupplierName", "SupplierTransactionID", "SupplierID", "PurchaseOrderID", "SupplierInvoiceNumber", "TransactionDate - Copy", "TransactionDate", "AmountExcludingTax", "TaxAmount", "FinalizationDate"}),
    #"Renamed Columns" = Table.RenameColumns(#"Reordered Columns",{{"TransactionDate - Copy", "TransactionYear"}}),
    #"Added Custom" = Table.AddColumn(#"Renamed Columns", "TotalAmount", each [AmountExcludingTax]+[TaxAmount]),
    #"Changed Type1" = Table.TransformColumnTypes(#"Added Custom",{{"TotalAmount", Currency.Type}}),
    #"Removed Other Columns" = Table.SelectColumns(#"Changed Type1",{"SupplierName", "SupplierTransactionID", "SupplierID", "PurchaseOrderID", "SupplierInvoiceNumber", "TransactionYear", "TransactionDate", "FinalizationDate", "TotalAmount"}),
    #"Added Custom1" = Table.AddColumn(#"Removed Other Columns", "Days", each [TransactionDate]-[FinalizationDate]),
    #"Changed Type2" = Table.TransformColumnTypes(#"Added Custom1",{{"Days", Int64.Type}}),
    #"Calculated Absolute Value" = Table.TransformColumns(#"Changed Type2",{{"Days", Number.Abs, Int64.Type}}),
    #"Added Conditional Column" = Table.AddColumn(#"Calculated Absolute Value", "OverUnder", each if [Days] >= 3 then "3 Days or More" else "2 Days or Less")
in
    #"Added Conditional Column"
