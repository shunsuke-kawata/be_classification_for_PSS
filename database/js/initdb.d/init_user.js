db = db.getSiblingDB("pss_mongo_db");

db.createUser({
  user: "user",
  pwd: "user",
  roles: [
    {
      role: "readWrite",
      db: "pss_mongo_db",
    },
  ],
});
