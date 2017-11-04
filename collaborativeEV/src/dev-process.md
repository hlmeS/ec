# meowcooler - development documentation

## Getting started
Let's create our new application by entering

```sh
mix phoenix.new meow-cooler

```
and following the instructions.
* configure the DB in config/dev.exs
  * db: *meowcooler_dev*
  * user: *meow*
  * pass: *Tyg3rme0w*
* Run ```mix ecto.create```

Some `psql` in here to create a new role.

```sql
CREATE ROLE meow WITH CREATEDB CREATEROLE LOGIN PASSWORD 'Tyg3rme0w';
```

# Creating models

Models are going to describe our DB schema, so we're just going to implement this based on the developed [conceptual](#) and [relational](#) DB model (v1.0).

The underlying DB is a Postgres DB and the DB connector is Ecto (default for Phoenix). Without an existing DB in place, we can use ecto migrations to create, alter, drop tables.

## changeset

``` elixir
defmodule Meowcooler.Repo.Migrations.AddCompanyTable do
  use Ecto.Migration

  def change do
    create table (:companies) do
      add :name, :string, unique: true
      timestamps()
    end
    create unique_index(:companies, [:name])
  end
end

defmodule Meowcooler.Repo.Migrations.AddUserTable do
  use Ecto.Migration

  def change do
    create table(:users) do
      add :username, :string, unique: true

      timestamps()
    end

    create unique_index(:users, [:username])
  end
end

defmodule Meowcooler.Repo.Migrations.Add_Container_Table do
  use Ecto.Migration

  def change do
    create table(:containers) do
      add :name, :string
      add :companyID, references(:companies, type: :int, column: :id)
    end

    alter table(:users) do
      add :companyID, references(:companies, type: :int, column: :id)
    end
  end

end
```

## Schema
``` elixir
defmodule Meowcooler.User do
  @moduledoc """
  Users of our system.
  """

  use Meowcooler.Web, :model

  schema "users" do
    field :username, :string

    belongs_to :company, Meowcooler.Company
    timestamps()
  end

  @doc """
  Builds a changeset based on the `struct` and `params`.
  """
  def changeset(struct, params \\ %{}) do
    struct
    |> cast(params, [:username])
    |> validate_required([:username])
    |> unique_constraint(:username)
  end
end

defmodule Meowcooler.Company do
  use Meowcooler.Web, :model

  schema "companies" do
    field :name, :string

    has_many :users, Meowcooler.User
    has_many :containers, Meowcooler.Container

    timestamps()
  end

  def changeset(model, params \\ %{}) do
    model
    |> cast(params, [:names])
    |> validate_required([:name])
    |> unique_constraint([:name])
  end

end
```

Next, add `cast_assoc`
