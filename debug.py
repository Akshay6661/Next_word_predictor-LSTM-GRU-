Client Brand = 
SWITCH(
    TRUE(),
    Dim_Quality_Client[Client_Name] IN {
        "Pfizer GO DSU", "Pfizer PSSR"
    }, "Pfizer",
    Dim_Quality_Client[Client_Name] IN {
        "Vantive AR", "Vantive CP"
    }, "Vantive",
    Dim_Quality_Client[Client_Name] IN {
        "GE-HC AR", "GE-HC ICSR"
    }, "GE",
    Dim_Quality_Client[Client_Name] IN {
        "MICC-Glenmark", "MICC-Piramal", "MICC-Cipla"
    }, "MICC",
    Dim_Quality_Client[Client_Name]
)
