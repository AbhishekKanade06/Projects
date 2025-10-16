# Hotel Management System

## Overview

The **Hotel Management System** is a Java-based application designed to streamline core hotel operations such as customer registration, room allocation, service management, billing, and checkout. It uses a MySQL database to store and retrieve customer and booking information, ensuring efficient management of hotel resources.

## Components

### 1. **Hotel.java**
- **Description:** The main entry point of the system containing the `main` method.
- **Responsibilities:**
  - Manages customer registration, information updates, viewing details, and checkout.
  - Provides a menu-driven interface for hotel staff or users to interact with the system.
  - Handles database connections and executes SQL queries for inserting, updating, and fetching customer and booking records.
  - Coordinates with `Room` and `Services` classes for room assignments and service additions.

### 2. **Room.java**
- **Description:** Extends `Hotel` class and manages room-related operations.
- **Responsibilities:**
  - Allows users to select a room type (AC or Non-AC).
  - Checks availability of rooms of the selected type from the database.
  - Assigns room numbers to customers.
  - Updates room status (occupied or available) in the database.
  - Updates customer records with the assigned room details and check-in status.

### 3. **Services.java**
- **Description:** Extends `Hotel` class and manages additional hotel services like Food, WiFi, Gym, Laundry, and Car Rent.
- **Responsibilities:**
  - Displays a menu of available services and their respective prices.
  - Records selected services for each customer.
  - Calculates and generates bills based on utilized services, including dates of service and total prices.
  - Fetches and displays detailed billing information.
  - Updates billing records in the database.

## How It Works

1. **Start the System:** The user begins at `Hotel.main()`, where they can choose options like register a customer, view details, update records, or check out.
2. **Customer Registration:** When registering a customer, the system collects personal details and ID proof, inserts this information into the customer database, and proceeds to room selection.
3. **Room Allocation:** The `Room` class handles room type selection (AC/Non-AC), checks room availability, and assigns a room to the customer.
4. **Service Selection:** The `Services` class allows customers to select additional services (e.g., food, WiFi) and adds them to the customer’s billing record.
5. **Checkout Process:** Upon checkout, the system finalizes the billing information, resets room and service statuses, and generates a detailed bill for the customer.

The system uses **JDBC** to interact with a MySQL database named **hotel** for data storage.

## Key Technologies & Libraries

- **Java (JDK 8+)**: The programming language used for implementation.
- **JDBC API**: For connecting to and interacting with the MySQL database.
- **MySQL**: The backend database for storing customer and room information.
- **Java Scanner**: Used to take user input via the console.
- **SQL Queries**: Utilized for CRUD operations (SELECT, INSERT, UPDATE).
- **Java Time API**: Used for handling dates (for services and billing).

## Usage

### Prerequisites

- Ensure that the **MySQL server** is running.
- Create the required database and table schema in MySQL (tables for `customers`, `rooms`, `services`, `bills`, etc.).

### Running the System

1. **Compile the Java files**:
   ```bash
   javac Hotel.java Room.java Services.java
