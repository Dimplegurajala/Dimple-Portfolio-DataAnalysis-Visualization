DELIMITER $$
CREATE PROCEDURE CreateIndex_Album_ArtistId()
BEGIN
    CREATE INDEX IFK_AlbumArtistId ON Album (ArtistId);
END$$
DELIMITER ;

-- --- Procedure for Customer-SupportRep Index ---
DELIMITER $$
CREATE PROCEDURE CreateIndex_Customer_SupportRepId()
BEGIN
    CREATE INDEX IFK_CustomerSupportRepId ON Customer (SupportRepId);
END$$
DELIMITER ;

-- --- Procedure for Employee-ReportsTo Index ---
DELIMITER $$
CREATE PROCEDURE CreateIndex_Employee_ReportsTo()
BEGIN
    CREATE INDEX IFK_EmployeeReportsTo ON Employee (ReportsTo);
END$$
DELIMITER ;

-- --- Procedure for Invoice-Customer Index ---
DELIMITER $$
CREATE PROCEDURE CreateIndex_Invoice_CustomerId()
BEGIN
    CREATE INDEX IFK_InvoiceCustomerId ON Invoice (CustomerId);
END$$
DELIMITER ;

-- --- Procedure for InvoiceLine-Invoice Index ---
DELIMITER $$
CREATE PROCEDURE CreateIndex_InvoiceLine_InvoiceId()
BEGIN
    CREATE INDEX IFK_InvoiceLineInvoiceId ON InvoiceLine (InvoiceId);
END$$
DELIMITER ;

-- --- Procedure for InvoiceLine-Track Index ---
DELIMITER $$
CREATE PROCEDURE CreateIndex_InvoiceLine_TrackId()
BEGIN
    CREATE INDEX IFK_InvoiceLineTrackId ON InvoiceLine (TrackId);
END$$
DELIMITER ;

-- --- Procedure for PlaylistTrack-Track Index ---
DELIMITER $$
CREATE PROCEDURE CreateIndex_PlaylistTrack_TrackId()
BEGIN
    CREATE INDEX IFK_PlaylistTrackTrackId ON PlaylistTrack (TrackId);
END$$
DELIMITER ;

-- --- Procedure for Track-Album Index ---
DELIMITER $$
CREATE PROCEDURE CreateIndex_Track_AlbumId()
BEGIN
    CREATE INDEX IFK_TrackAlbumId ON Track (AlbumId);
END$$
DELIMITER ;

-- --- Procedure for Track-Genre Index ---
DELIMITER $$
CREATE PROCEDURE CreateIndex_Track_GenreId()
BEGIN
    CREATE INDEX IFK_TrackGenreId ON Track (GenreId);
END$$
DELIMITER ;

-- --- Procedure for Track-MediaType Index ---
DELIMITER $$
CREATE PROCEDURE CreateIndex_Track_MediaTypeId()
BEGIN
    CREATE INDEX IFK_TrackMediaTypeId ON Track (MediaTypeId);
END$$
DELIMITER ;

SELECT *
FROM Artist
WHERE Name = 'AC/DC';

SELECT
    Artist.Name,
    COUNT(Album.AlbumId) AS NumberOfAlbums
FROM
    Artist
JOIN
    Album ON Artist.ArtistId = Album.ArtistId
GROUP BY
    Artist.Name;
    
SELECT
    Album.Title AS AlbumTitle,
    Artist.Name AS ArtistName
FROM
    Album
JOIN
    Artist ON Album.ArtistId = Artist.ArtistId;

SELECT
    Name,
    Composer,
    Milliseconds
FROM
    Track
ORDER BY
    Milliseconds DESC
LIMIT 10;