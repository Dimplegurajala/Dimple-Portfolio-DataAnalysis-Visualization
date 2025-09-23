SET foreign_key_checks = 0; -- Temporarily disable checks to allow truncation.
TRUNCATE TABLE Employee;
TRUNCATE TABLE Customer;
TRUNCATE TABLE Artist;
TRUNCATE TABLE Album;
TRUNCATE TABLE playlist;
TRUNCATE TABLE track;
TRUNCATE TABLE playlisttrack;
TRUNCATE TABLE invoice;
TRUNCATE TABLE invoiceline;
-- Add a TRUNCATE TABLE line for every table you have data for.
SET foreign_key_checks = 1; 
SET foreign_key_checks = 0;

SET foreign_key_checks = 1; 




