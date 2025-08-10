/* Project-Dimple */


Create database PROJECT;

USE PROJECT;

CREATE TABLE Categories (
    category_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL
);

CREATE TABLE Items (
    item_id INT AUTO_INCREMENT PRIMARY KEY,
    category_id INT,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    price DECIMAL(10, 2),
    shelf_life INT,
    FOREIGN KEY (category_id) REFERENCES Categories(category_id)
);

CREATE TABLE Inventory (
    inventory_id INT AUTO_INCREMENT PRIMARY KEY,
    item_id INT,
    quantity INT NOT NULL,
    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
    minimum_required INT,
    FOREIGN KEY (item_id) REFERENCES Items(item_id)
);

CREATE TABLE Suppliers (
    supplier_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    contact_info TEXT,
    preferred BOOLEAN
);

CREATE TABLE Supply_Orders (
    order_id INT AUTO_INCREMENT PRIMARY KEY,
    supplier_id INT,
    order_date DATE,
    expected_delivery DATE,
    status VARCHAR(50),
    FOREIGN KEY (supplier_id) REFERENCES Suppliers(supplier_id)
);

CREATE TABLE Order_Items (
    order_item_id INT AUTO_INCREMENT PRIMARY KEY,
    order_id INT,
    item_id INT,
    quantity_ordered INT,
    FOREIGN KEY (order_id) REFERENCES Supply_Orders(order_id),
    FOREIGN KEY (item_id) REFERENCES Items(item_id)
);

/* Inserting values into the above Tables */

/* Project-Dimple */

/* Switch to the PROJECT database */
USE PROJECT;

/* Insert data into Categories table */
INSERT INTO Categories (name) VALUES 
('Toiletries'),
('Vegan Products'),
('Vegetarian Products'),
('Meat Products'),
('Daily Essentials');

/* Insert data into Items table */
INSERT INTO Items (category_id, name, description, price, shelf_life) VALUES 
(1, 'Shampoo', '500ml bottle of shampoo', 5.99, 730),
(1, 'Toothpaste', '100g tube of toothpaste', 2.99, 730),
(2, 'Almond Milk', '1 liter of almond milk', 3.49, 14),
(2, 'Vegan Cheese', '200g pack of vegan cheese', 4.99, 30),
(3, 'Vegetarian Sausages', 'Pack of 6 vegetarian sausages', 3.99, 14),
(4, 'Chicken Breast', '500g of chicken breast', 5.49, 7),
(4, 'Ground Beef', '1kg of ground beef', 7.99, 7),
(5, 'Milk', '1 liter of milk', 1.99, 7),
(5, 'Eggs', 'Dozen eggs', 2.49, 14),
(5, 'Bread', 'Loaf of whole grain bread', 2.99, 7);

/* Insert data into Inventory table */
INSERT INTO Inventory (item_id, quantity, minimum_required) VALUES 
(1, 100, 20),
(2, 150, 30),
(3, 50, 10),
(4, 40, 10),
(5, 30, 10),
(6, 20, 5),
(7, 10, 5),
(8, 60, 15),
(9, 80, 20),
(10, 50, 15);

/* Insert data into Suppliers table */
INSERT INTO Suppliers (name, contact_info, preferred) VALUES 
('Supplier A', '1234 Main St, Anytown, USA', TRUE),
('Supplier B', '5678 Market St, Othertown, USA', FALSE),
('Supplier C', '9101 Broadway Ave, Anycity, USA', TRUE);

/* Insert data into Supply_Orders table */
INSERT INTO Supply_Orders (supplier_id, order_date, expected_delivery, status) VALUES 
(1, '2024-05-01', '2024-05-05', 'ordered'),
(2, '2024-05-02', '2024-05-06', 'delivered'),
(3, '2024-05-03', '2024-05-07', 'ordered');

/* Insert data into Order_Items table */
INSERT INTO Order_Items (order_id, item_id, quantity_ordered) VALUES 
(1, 1, 50),
(1, 3, 20),
(2, 4, 40),
(2, 6, 10),
(3, 8, 30),
(3, 10, 20);
