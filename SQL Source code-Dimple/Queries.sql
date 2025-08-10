
USE PROJECT;

/* Total Stock */
SELECT SUM(quantity) AS total_stock FROM Inventory;

/* Items below minimum level */
SELECT * FROM Inventory WHERE quantity < minimum_required;

/* Supplier Orders */
SELECT * FROM Supply_Orders WHERE status = 'ordered';

/* Supplier Information */
SELECT * FROM Suppliers;

/* Order Details */
SELECT * FROM Order_Items JOIN Supply_Orders ON Order_Items.order_id = Supply_Orders.order_id;

/* Query to find all items in a specific category (e.g., 'Daily Essentials') */
SELECT Items.name, Items.description, Items.price
FROM Items
JOIN Categories ON Items.category_id = Categories.category_id
WHERE Categories.name = 'Daily Essentials';

/* Query to find the total value of items in inventory */
SELECT SUM(Inventory.quantity * Items.price) AS total_inventory_value
FROM Inventory
JOIN Items ON Inventory.item_id = Items.item_id;
