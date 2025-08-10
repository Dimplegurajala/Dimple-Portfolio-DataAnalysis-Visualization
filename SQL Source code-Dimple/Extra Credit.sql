/* Extra Credit */

Use PROJECT;

USE PROJECT;

/* Total Stock Value Calculation */
DELIMITER $$

DROP PROCEDURE IF EXISTS CalculateTotalStockValue $$

CREATE PROCEDURE CalculateTotalStockValue()
BEGIN
    SELECT SUM(quantity * price) AS total_stock_value
    FROM Inventory 
    JOIN Items ON Inventory.item_id = Items.item_id;
END $$

DELIMITER ;

/* List of Delivered Orders */
DELIMITER $$

DROP PROCEDURE IF EXISTS ListDeliveredOrders $$

CREATE PROCEDURE ListDeliveredOrders()
BEGIN
    SELECT * FROM Supply_Orders WHERE status = 'delivered';
END $$

DELIMITER ;
